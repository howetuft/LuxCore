// C++ port of BlendLuxCore's mesh_converter.compute_submeshes().
//
// Ported 1:1 from the numpy implementation (same semantics: weld keys,
// island union-find over triangles via sorted edge occurrences, small-island
// merge, (material, island) grouping, candidate sharing test, last-write-wins
// gathers, per-submesh compaction, transform baking). The heavy stages run
// with the GIL released and use TBB (parallel sorts, union-find, per-group
// payload loop).

#include "compute_submeshes.h"

#include <tbb/parallel_for.h>
#include <tbb/scalable_allocator.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <tuple>

#include "slg/utils/group_by_equivalence.h"

namespace {

constexpr size_t MIN_ISLAND_TRIANGLE_COUNT = 64;
// float(np.cos(np.deg2rad(30.0)))
constexpr double MIN_ISLAND_MERGE_COS_ANGLE = 0.86602540378443864676;

using Clock = std::chrono::steady_clock;

struct StageTimer {
	std::vector<std::pair<std::string, double>> stages;
	Clock::time_point t0 = Clock::now();
	void stage(const char *name) {
		auto now = Clock::now();
		stages.emplace_back(name, std::chrono::duration<double>(now - t0).count());
		t0 = now;
	}
};

//------------------------------------------------------------------------------
// Parallel argsort helpers (same semantics as pyluxcore's parallel_argsort:
// tbb::parallel_sort over (key, index) pairs, ties broken by index, so the
// result is deterministic)
//------------------------------------------------------------------------------

template <typename T>
std::vector<int64_t> argsort1(const T *keys, size_t n) {
	std::vector<std::pair<T, int64_t>> pairs(n);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
		[&](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i < r.end(); ++i)
				pairs[i] = std::make_pair(keys[i], static_cast<int64_t>(i));
		});
	tbb::parallel_sort(pairs.begin(), pairs.end());
	std::vector<int64_t> out(n);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
		[&](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i < r.end(); ++i)
				out[i] = pairs[i].second;
		});
	return out;
}

template <typename T, typename U>
std::vector<int64_t> argsort2(const T *keys, const U *secondary, size_t n) {
	std::vector<std::tuple<T, U, int64_t>> items(n);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
		[&](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i < r.end(); ++i)
				items[i] = std::make_tuple(keys[i], secondary[i], static_cast<int64_t>(i));
		});
	tbb::parallel_sort(items.begin(), items.end());
	std::vector<int64_t> out(n);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
		[&](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i < r.end(); ++i)
				out[i] = std::get<2>(items[i]);
		});
	return out;
}

std::vector<uint32_t> union_find_labels(size_t num_elements, const std::vector<slg::equiv::Relation> &rel) {
	return slg::equiv::GroupByEquivalenceLabels(
		num_elements, std::span<const slg::equiv::Relation>(rel));
}

// Serial union-find (path halving + union by size) with dense labels in
// ascending root order. Produces the same partition as
// slg::equiv::GroupByEquivalenceLabels, but without its per-reduce-body
// construction of a numElements-sized parent array, which dominates the cost
// when the relation count is small relative to the element count (the common
// case for mesh island graphs: ~1.5 relations per triangle).
std::vector<uint32_t> serial_union_find_labels(size_t num_elements, const std::vector<slg::equiv::Relation> &rel) {
	if (num_elements == 0)
		return {};
	std::vector<uint32_t> parent(num_elements);
	std::iota(parent.begin(), parent.end(), 0u);
	std::vector<uint32_t> sz(num_elements, 1);
	auto find = [&parent](size_t i) {
		while (parent[i] != i) {
			parent[i] = parent[parent[i]];
			i = parent[i];
		}
		return i;
	};
	for (const auto &r : rel) {
		size_t a = find(r.first), b = find(r.second);
		if (a == b)
			continue;
		if (sz[a] < sz[b])
			std::swap(a, b);
		parent[b] = (uint32_t)a;
		sz[a] += sz[b];
	}
	// Dense labels in ascending root order: mark the roots in a bitmap, then
	// prefix-sum it (O(n), no sort)
	std::vector<uint32_t> roots(num_elements);
	std::vector<uint8_t> isRoot(num_elements, 0);
	for (size_t i = 0; i < num_elements; ++i) {
		const uint32_t r = (uint32_t)find(i);
		roots[i] = r;
		isRoot[r] = 1;
	}
	std::vector<uint32_t> labelOfRoot(num_elements, 0);
	uint32_t count = 0;
	for (size_t i = 0; i < num_elements; ++i) {
		labelOfRoot[i] = count;
		count += isRoot[i];
	}
	std::vector<uint32_t> labels(num_elements);
	for (size_t i = 0; i < num_elements; ++i)
		labels[i] = labelOfRoot[roots[i]];
	return labels;
}

// Containers allocated per iteration inside tbb::parallel_for bodies (the
// per-group temporaries of the payload loop) use tbbmalloc: it serves each
// thread from its own cache, avoiding contention on glibc malloc's arenas.
// Same pattern as slg's group_by_equivalence.cpp. The big per-call buffers
// (weld, sig, candidates, attr arrays) are allocated once per call, outside
// the parallel regions; attr buffers larger than the mmap threshold go
// straight to mmap anyway.
template <typename T>
using ScalableVector = std::vector<T, tbb::scalable_allocator<T>>;

//------------------------------------------------------------------------------
// Weld: order-preserving float-bit keys, (x,y) packed into a uint64 sort key
// with z as secondary, -0.0 mapped to key 0 (same as the numpy version)
//------------------------------------------------------------------------------

inline uint32_t sortable_float_bits(float f) {
	uint32_t b;
	std::memcpy(&b, &f, 4);
	uint32_t key = (b >> 31) ? ~b : (b ^ 0x80000000u);
	if (b == 0x80000000u)
		key = 0;  // -0.0
	return key;
}

struct WeldResult {
	std::vector<float> points;         // [P, 3]
	std::vector<int64_t> first_index;  // [P]
	std::vector<int64_t> inverse;      // [V]
};

// NaN-aware exact weld (fallback path, mirrors np.unique(axis=0) semantics:
// NaNs compare equal to each other and sort last)
WeldResult weld_points_nan_fallback(const float *vp, size_t V) {
	struct Cmp {
		const float *vp;
		bool operator()(int64_t a, int64_t b) const {
			for (int c = 0; c < 3; ++c) {
				float x = vp[a * 3 + c], y = vp[b * 3 + c];
				bool anx = std::isnan(x), bny = std::isnan(y);
				if (anx && bny) continue;
				if (anx) return false;  // NaN sorts last
				if (bny) return true;
				if (x != y) return x < y;
			}
			return a < b;
		}
	};
	std::vector<int64_t> order(V);
	for (size_t i = 0; i < V; ++i) order[i] = (int64_t)i;
	std::sort(order.begin(), order.end(), Cmp{vp});
	WeldResult res;
	res.inverse.assign(V, 0);
	int64_t group = -1;
	for (size_t i = 0; i < V; ++i) {
		if (i == 0 || Cmp{vp}(order[i - 1], order[i])) {
			++group;
			res.first_index.push_back(order[i]);
		}
		res.inverse[order[i]] = group;
	}
	res.points.resize(res.first_index.size() * 3);
	for (size_t p = 0; p < res.first_index.size(); ++p)
		std::memcpy(&res.points[p * 3], &vp[res.first_index[p] * 3], 12);
	return res;
}

WeldResult weld_points(const float *vp, size_t V) {
	if (V == 0)
		return {};
	bool has_nan = tbb::parallel_reduce(
		tbb::blocked_range<size_t>(0, V),
		false,
		[&](const tbb::blocked_range<size_t> &r, bool init) {
			for (size_t i = r.begin(); i < r.end(); ++i)
				if (std::isnan(vp[i * 3]) || std::isnan(vp[i * 3 + 1]) || std::isnan(vp[i * 3 + 2]))
					return true;
			return init;
		},
		[](bool a, bool b) { return a || b; });
	if (has_nan)
		return weld_points_nan_fallback(vp, V);

	std::vector<uint32_t> kx(V), ky(V), kz(V);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, V),
		[&](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i < r.end(); ++i) {
				kx[i] = sortable_float_bits(vp[i * 3]);
				ky[i] = sortable_float_bits(vp[i * 3 + 1]);
				kz[i] = sortable_float_bits(vp[i * 3 + 2]);
			}
		});
	std::vector<uint64_t> key64(V);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, V),
		[&](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i < r.end(); ++i)
				key64[i] = (static_cast<uint64_t>(kx[i]) << 32) | ky[i];
		});
	std::vector<int64_t> order = argsort2<uint64_t, uint32_t>(key64.data(), kz.data(), V);

	WeldResult res;
	res.inverse.assign(V, 0);
	std::vector<char> boundary(V);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, V),
		[&](const tbb::blocked_range<size_t> &r) {
			for (size_t i = r.begin(); i < r.end(); ++i) {
				if (i == 0)
					boundary[i] = 1;
				else {
					int64_t a = order[i - 1], b = order[i];
					boundary[i] = !(key64[a] == key64[b] && kz[a] == kz[b]);
				}
			}
		});
	int64_t num_points = 0;
	for (size_t i = 0; i < V; ++i)
		num_points += boundary[i];
	res.first_index.resize(num_points);
	res.points.resize(num_points * 3);
	int64_t group = -1;
	for (size_t i = 0; i < V; ++i) {
		if (boundary[i]) {
			++group;
			res.first_index[group] = order[i];
			std::memcpy(&res.points[group * 3], &vp[order[i] * 3], 12);
		}
		res.inverse[order[i]] = group;
	}
	return res;
}

//------------------------------------------------------------------------------
// 3x3 helpers for the transform baking
//------------------------------------------------------------------------------

double det3(const float m[9]) {
	return (double)m[0] * ((double)m[4] * m[8] - (double)m[5] * m[7])
		- (double)m[1] * ((double)m[3] * m[8] - (double)m[5] * m[6])
		+ (double)m[2] * ((double)m[3] * m[7] - (double)m[4] * m[6]);
}

void inv3(const float m[9], float out[9]) {
	double d = det3(m);
	double id = 1.0 / d;
	out[0] = (float)(id * ((double)m[4] * m[8] - (double)m[5] * m[7]));
	out[1] = (float)(id * ((double)m[2] * m[7] - (double)m[1] * m[8]));
	out[2] = (float)(id * ((double)m[1] * m[5] - (double)m[2] * m[4]));
	out[3] = (float)(id * ((double)m[5] * m[6] - (double)m[3] * m[8]));
	out[4] = (float)(id * ((double)m[0] * m[8] - (double)m[2] * m[6]));
	out[5] = (float)(id * ((double)m[2] * m[3] - (double)m[0] * m[5]));
	out[6] = (float)(id * ((double)m[3] * m[7] - (double)m[4] * m[6]));
	out[7] = (float)(id * ((double)m[1] * m[6] - (double)m[0] * m[7]));
	out[8] = (float)(id * ((double)m[0] * m[4] - (double)m[1] * m[3]));
}

//------------------------------------------------------------------------------
// Output structures
//------------------------------------------------------------------------------

struct ScalableFree {
	void operator()(float *p) const { scalable_free(p); }
};

struct AttrRef {
	int cand = -1;                  // >= 0: shared candidate index, -1: owned buffer
	// [P, k] row-major, only when cand < 0. Scalable allocation: the payload
	// loop runs in parallel over groups and the default malloc would contend
	std::unique_ptr<float[], ScalableFree> owned;
	size_t owned_size = 0;
};

struct SubmeshOut {
	char name[64];                    // fixed buffer: no heap alloc per group
	bool points_is_name = false;      // share mode, not the first submesh
	char points_name[64];             // name of the first submesh (share mode)
	bool points_shared = false;       // share mode, first submesh (uses welded points)
	std::vector<float> points_owned;  // compact mode
	std::vector<uint32_t> triangles;  // [g, 3] flat
	size_t n_points = 0;              // P of the attr arrays
	AttrRef normals;
	std::vector<AttrRef> uvs, colors, alphas;
	bool has_transform = false;
	float transform[16];
	bool has_creases = false;  // only the first submesh carries them (shared arrays in R)
	bool normals_pretransformed = false;
	uint32_t mat = 0;
	int slot = 0;
};

struct ComputeResult {
	std::vector<SubmeshOut> submeshes;
	std::vector<uint32_t> crease_pairs;    // flat [2E], shared by all submeshes
	std::vector<float> crease_weights;     // [E]
	bool has_creases = false;

	std::vector<std::vector<float>> candidates;
	std::vector<int> candidate_cols;  // floats per row of each candidate
	std::vector<float> welded_points;
	std::vector<int64_t> island_debug;  // per-triangle island labels (A-B tests)
	StageTimer timer;
	size_t num_points = 0;
	size_t num_raw = 0;
	double duration = 0.0;
};

// Zero-copy wrap of an owned vector into a numpy array (the capsule keeps the
// vector alive as long as the array exists)
template <typename T>
py::array_t<T> wrap_owned(std::vector<T> &&vec, std::vector<py::ssize_t> shape) {
	auto *holder = new std::vector<T>(std::move(vec));
	py::capsule cap(holder, [](void *p) { delete static_cast<std::vector<T> *>(p); });
	std::vector<py::ssize_t> strides(shape.size());
	py::ssize_t s = sizeof(T);
	for (size_t i = shape.size(); i-- > 0;) {
		strides[i] = s;
		s *= shape[i];
	}
	return py::array_t<T>(shape, strides, holder->data(), cap);
}

// Same as wrap_owned, but for an uninitialized buffer (rows that the submesh
// never references are left untouched -- they are never read, exactly like
// the numpy fallback's np.empty)
template <typename T, typename Buf>
py::array_t<T> wrap_owned_buf(Buf &&buf, std::vector<py::ssize_t> shape) {
	T *data = buf.get();
	using BufT = std::decay_t<Buf>;
	auto *holder = new BufT(std::move(buf));
	py::capsule cap(holder, [](void *p) { delete static_cast<BufT *>(p); });
	std::vector<py::ssize_t> strides(shape.size());
	py::ssize_t s = sizeof(T);
	for (size_t i = shape.size(); i-- > 0;) {
		strides[i] = s;
		s *= shape[i];
	}
	return py::array_t<T>(shape, strides, data, cap);
}

}  // namespace

py::tuple compute_submeshes_c(
	py::array_t<float, py::array::c_style | py::array::forcecast> vertex_points,
	py::array_t<uint32_t, py::array::c_style | py::array::forcecast> loop_vertices,
	py::array_t<float, py::array::c_style | py::array::forcecast> loop_normals,
	py::array_t<uint32_t, py::array::c_style | py::array::forcecast> triangle_loops,
	py::array_t<uint32_t, py::array::c_style | py::array::forcecast> loop_triangle_materials,
	std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> loop_uvs,
	std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> color_attrs,
	std::vector<std::string> color_domains,
	py::object edge_verts,
	py::object edge_crease,
	py::object transform,
	bool is_viewport_render,
	bool use_instancing,
	bool share_points,
	std::string mesh_key)
{
	// ---- validate shapes and grab raw pointers (GIL held) ----
	if (vertex_points.ndim() != 2 || vertex_points.shape(1) != 3)
		throw std::runtime_error("compute_submeshes_c: vertex_points must be [V,3] float32");
	if (loop_normals.ndim() != 2 || loop_normals.shape(1) != 3)
		throw std::runtime_error("compute_submeshes_c: loop_normals must be [L,3] float32");
	if (triangle_loops.ndim() != 2 || triangle_loops.shape(1) != 3)
		throw std::runtime_error("compute_submeshes_c: triangle_loops must be [T,3] uint32");

	const size_t V = vertex_points.shape(0);
	const size_t L = loop_vertices.shape(0);
	const size_t T = triangle_loops.shape(0);
	if (loop_normals.shape(0) != L)
		throw std::runtime_error("compute_submeshes_c: loop_normals/loop_vertices length mismatch");
	if (loop_triangle_materials.shape(0) != T)
		throw std::runtime_error("compute_submeshes_c: loop_triangle_materials length mismatch");
	const size_t n_uv = loop_uvs.size();
	for (size_t i = 0; i < n_uv; ++i) {
		if (loop_uvs[i].ndim() != 2 || loop_uvs[i].shape(0) != L || loop_uvs[i].shape(1) != 2)
			throw std::runtime_error("compute_submeshes_c: loop_uvs layers must be [L,2] float32");
	}
	if (color_attrs.size() != color_domains.size())
		throw std::runtime_error("compute_submeshes_c: color_attrs/color_domains length mismatch");

	const float *vp = vertex_points.data();
	const uint32_t *lv = loop_vertices.data();
	float *ln = loop_normals.mutable_data();
	const uint32_t *tl = triangle_loops.data();
	const uint32_t *mats = loop_triangle_materials.data();
	std::vector<const float *> uv_ptrs(n_uv);
	for (size_t i = 0; i < n_uv; ++i)
		uv_ptrs[i] = loop_uvs[i].data();
	std::vector<const float *> col_ptrs(color_attrs.size());
	std::vector<size_t> col_rows(color_attrs.size());
	for (size_t i = 0; i < color_attrs.size(); ++i) {
		if (color_attrs[i].ndim() != 2 || color_attrs[i].shape(1) != 4)
			throw std::runtime_error("compute_submeshes_c: color_attrs layers must be [N,4] float32");
		col_ptrs[i] = color_attrs[i].data();
		col_rows[i] = color_attrs[i].shape(0);
	}

	const int32_t *ev = nullptr;
	const float *ec = nullptr;
	size_t E = 0;
	bool has_edge_verts = !edge_verts.is_none();
	if (has_edge_verts) {
		auto ev_arr = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(edge_verts);
		if (ev_arr.ndim() != 2 || ev_arr.shape(1) != 2)
			throw std::runtime_error("compute_submeshes_c: edge_verts must be [E,2] int32");
		E = ev_arr.shape(0);
		ev = ev_arr.data();
		if (!edge_crease.is_none()) {
			auto ec_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(edge_crease);
			if ((size_t)ec_arr.shape(0) != E)
				throw std::runtime_error("compute_submeshes_c: edge_crease length mismatch");
			ec = ec_arr.data();
		}
	}

	float xform[16] = {0};
	bool has_xform = !transform.is_none();
	if (has_xform) {
		auto t_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(transform);
		if (t_arr.ndim() != 2 || t_arr.shape(0) != 4 || t_arr.shape(1) != 4)
			throw std::runtime_error("compute_submeshes_c: transform must be [4,4] float32");
		std::memcpy(xform, t_arr.data(), sizeof(xform));
	}

	// ---- compute (GIL released) ----
	ComputeResult R;
	{
		Py_BEGIN_ALLOW_THREADS
		auto wall0 = Clock::now();

		// --- weld ---
		WeldResult W = weld_points(vp, V);
		R.welded_points = std::move(W.points);
		const std::vector<int64_t> first_index = std::move(W.first_index);
		const std::vector<int64_t> weld_inverse = std::move(W.inverse);
		const size_t P = R.welded_points.size() / 3;
		R.num_points = P;
		R.num_raw = V;
		R.timer.stage("weld");

		// --- per-loop signature: uv layers + shading normal ---
		const size_t K = 2 * n_uv + 3;
		std::vector<float> sig(L * K);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, L),
			[&](const tbb::blocked_range<size_t> &r) {
				for (size_t l = r.begin(); l < r.end(); ++l) {
					float *dst = &sig[l * K];
					for (size_t u = 0; u < n_uv; ++u) {
						dst[2 * u] = uv_ptrs[u][l * 2];
						dst[2 * u + 1] = uv_ptrs[u][l * 2 + 1];
					}
					dst[2 * n_uv] = ln[l * 3];
					dst[2 * n_uv + 1] = ln[l * 3 + 1];
					dst[2 * n_uv + 2] = ln[l * 3 + 2];
				}
			});

		// --- edge creases (remapped to welded point ids, deduped) ---
		// Python: keep = crease > 0; keys over kept edges; np.unique(keys,
		// return_index=True) -> first kept occurrence per unique key, ordered
		// by sorted key
		std::vector<uint32_t> crease_pairs;
		std::vector<float> crease_weights;
		if (has_edge_verts) {
			std::vector<uint64_t> keys;
			std::vector<int64_t> w0, w1;
			std::vector<float> cw;
			keys.reserve(E);
			for (size_t e = 0; e < E; ++e) {
				if (!ec || !(ec[e] > 0.0f))
					continue;
				int64_t a = weld_inverse[ev[e * 2]];
				int64_t b = weld_inverse[ev[e * 2 + 1]];
				uint64_t lo = (uint64_t)std::min(a, b), hi = (uint64_t)std::max(a, b);
				keys.push_back((lo << 32) | hi);
				w0.push_back(a);
				w1.push_back(b);
				cw.push_back(ec[e]);
			}
			const size_t nk = keys.size();
			std::vector<int64_t> order = nk ? argsort1<uint64_t>(keys.data(), nk) : std::vector<int64_t>();
			std::vector<size_t> first;
			for (size_t i = 0; i < nk; ++i)
				if (i == 0 || keys[order[i]] != keys[order[i - 1]])
					first.push_back((size_t)order[i]);
			// 'first' is already in sorted-key order; np.unique(return_index)
			// gives the first occurrence per unique value which, for a stable
			// sort, is the first element of each run
			crease_pairs.resize(first.size() * 2);
			crease_weights.resize(first.size());
			for (size_t i = 0; i < first.size(); ++i) {
				size_t e = first[i];
				crease_pairs[2 * i] = (uint32_t)w0[e];
				crease_pairs[2 * i + 1] = (uint32_t)w1[e];
				float w = cw[e] * 10.0f;
				crease_weights[i] = w > 10.0f ? 10.0f : w;
			}
		}
		R.crease_pairs = std::move(crease_pairs);
		R.crease_weights = std::move(crease_weights);
		R.has_creases = has_edge_verts;
		R.timer.stage("creases");

		// --- edge occurrences ---
		const size_t n_occ = T * 3;
		std::vector<int64_t> loop_welded(L);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, L),
			[&](const tbb::blocked_range<size_t> &r) {
				for (size_t l = r.begin(); l < r.end(); ++l)
					loop_welded[l] = weld_inverse[lv[l]];
			});
		std::vector<int64_t> e_key(n_occ), e_lo(n_occ), e_hi(n_occ), l_lo(n_occ), l_hi(n_occ);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, T),
			[&](const tbb::blocked_range<size_t> &r) {
				for (size_t t = r.begin(); t < r.end(); ++t) {
					// edges (c0,c1), (c1,c2), (c2,c0) -- same layout as
					// np.stack([tl, np.roll(tl,-1,axis=1)], axis=2).reshape(-1,2)
					for (int j = 0; j < 3; ++j) {
						size_t occ = t * 3 + j;
						int64_t la = tl[t * 3 + j];
						int64_t lb = tl[t * 3 + (j + 1) % 3];
						int64_t wa = loop_welded[la], wb = loop_welded[lb];
						if (wa > wb) {
							e_lo[occ] = wb; e_hi[occ] = wa;
							l_lo[occ] = lb; l_hi[occ] = la;
						} else {
							e_lo[occ] = wa; e_hi[occ] = wb;
							l_lo[occ] = la; l_hi[occ] = lb;
						}
						e_key[occ] = e_lo[occ] * (int64_t)P + e_hi[occ];
					}
				}
			});

		std::vector<int64_t> order = T ? argsort1<int64_t>(e_key.data(), n_occ) : std::vector<int64_t>();
		std::vector<int64_t> k_s(n_occ);
		if (T) {
			tbb::parallel_for(tbb::blocked_range<size_t>(0, n_occ),
				[&](const tbb::blocked_range<size_t> &r) {
					for (size_t i = r.begin(); i < r.end(); ++i)
						k_s[i] = e_key[order[i]];
				});
		}
		R.timer.stage("edge_sort");

		// triangle of each loop / occurrence
		std::vector<int64_t> tri_of_loop(L, 0);
		std::vector<int64_t> tri_of_sorted(n_occ);
		if (T) {
			tbb::parallel_for(tbb::blocked_range<size_t>(0, T),
				[&](const tbb::blocked_range<size_t> &r) {
					for (size_t t = r.begin(); t < r.end(); ++t) {
						for (int j = 0; j < 3; ++j) {
							tri_of_sorted[t * 3 + j] = (int64_t)t;
							tri_of_loop[tl[t * 3 + j]] = (int64_t)t;
						}
					}
				});
			std::vector<int64_t> tmp = tri_of_sorted;
			tbb::parallel_for(tbb::blocked_range<size_t>(0, n_occ),
				[&](const tbb::blocked_range<size_t> &r) {
					for (size_t i = r.begin(); i < r.end(); ++i)
						tri_of_sorted[i] = tmp[order[i]];
				});
		}

		// union pairs over triangles across shared edges with matching sigs
		std::vector<slg::equiv::Relation> pairs;
		if (T) {
			pairs.reserve(n_occ / 2 + 1);
			std::vector<char> join(n_occ, 0);
			const size_t sig_bytes = K * 4;
			tbb::parallel_for(tbb::blocked_range<size_t>(1, n_occ),
				[&](const tbb::blocked_range<size_t> &r) {
					for (size_t i = r.begin(); i < r.end(); ++i) {
						if (k_s[i] != k_s[i - 1])
							continue;
						const float *sa = &sig[l_lo[order[i]] * K];
						const float *sb = &sig[l_lo[order[i - 1]] * K];
						const float *ha = &sig[l_hi[order[i]] * K];
						const float *hb = &sig[l_hi[order[i - 1]] * K];
						if (std::memcmp(sa, sb, sig_bytes) == 0 && std::memcmp(ha, hb, sig_bytes) == 0)
							join[i] = 1;
					}
				});
			for (size_t i = 1; i < n_occ; ++i)
				if (join[i])
					pairs.emplace_back((uint32_t)tri_of_sorted[i], (uint32_t)tri_of_sorted[i - 1]);
		}
		std::vector<int64_t> island;
		if (T) {
			std::vector<uint32_t> labels = serial_union_find_labels(T, pairs);
			island.assign(labels.begin(), labels.end());
		}
		R.timer.stage("union_find");

		// --- merge very small islands into an adjacent island ---
		if (T) {
			int64_t n_islands0 = 0;
			for (size_t t = 0; t < T; ++t)
				n_islands0 = std::max(n_islands0, island[t] + 1);
			std::vector<int64_t> island_size(n_islands0, 0);
			for (size_t t = 0; t < T; ++t)
				++island_size[island[t]];
			std::vector<char> small(n_islands0);
			bool any_small = false;
			for (int64_t i = 0; i < n_islands0; ++i) {
				small[i] = island_size[i] < (int64_t)MIN_ISLAND_TRIANGLE_COUNT;
				any_small |= small[i];
			}
			if (any_small) {
				std::vector<int64_t> starts;
				for (size_t i = 0; i < n_occ; ++i)
					if (i == 0 || k_s[i] != k_s[i - 1])
						starts.push_back((int64_t)i);
				std::vector<int64_t> ends;
				for (size_t i = 1; i < starts.size(); ++i)
					ends.push_back(starts[i] - 1);
				ends.push_back((int64_t)n_occ - 1);
				const size_t n_runs = starts.size();
				std::vector<int64_t> isl_a(n_runs), isl_b(n_runs);
				std::vector<char> adj(n_runs, 0), not_hard(n_runs, 0), any_s(n_runs, 0);
				tbb::parallel_for(tbb::blocked_range<size_t>(0, n_runs),
					[&](const tbb::blocked_range<size_t> &r) {
						for (size_t i = r.begin(); i < r.end(); ++i) {
							int64_t run_len = ends[i] - starts[i] + 1;
							int64_t a = island[tri_of_sorted[starts[i]]];
							int64_t b = island[tri_of_sorted[ends[i]]];
							isl_a[i] = a;
							isl_b[i] = b;
							if (run_len < 2 || a == b)
								continue;
							adj[i] = 1;
							const float *n1 = &ln[l_lo[order[starts[i]]] * 3];
							const float *n2 = &ln[l_hi[order[ends[i]]] * 3];
							float cosang = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
							not_hard[i] = cosang > MIN_ISLAND_MERGE_COS_ANGLE;
							any_s[i] = small[a] || small[b];
						}
					});
				std::vector<std::pair<uint32_t, uint32_t>> isl_pairs;
				for (size_t i = 0; i < n_runs; ++i)
					if (adj[i] && any_s[i] && not_hard[i])
						isl_pairs.emplace_back((uint32_t)isl_a[i], (uint32_t)isl_b[i]);
				if (!isl_pairs.empty()) {
					// np.unique(axis=0): lexicographic sort + dedupe
					std::sort(isl_pairs.begin(), isl_pairs.end());
					isl_pairs.erase(std::unique(isl_pairs.begin(), isl_pairs.end()), isl_pairs.end());
					std::vector<slg::equiv::Relation> rel;
					rel.reserve(isl_pairs.size());
					for (auto &p : isl_pairs)
						rel.emplace_back(p.first, p.second);
					std::vector<uint32_t> isl_map = serial_union_find_labels((size_t)n_islands0, rel);
					for (size_t t = 0; t < T; ++t)
						island[t] = isl_map[island[t]];
				}
			}
		}
		R.timer.stage("islands");
		R.island_debug = island;

		// loop island
		std::vector<int64_t> loop_island(L);
		if (T) {
			tbb::parallel_for(tbb::blocked_range<size_t>(0, L),
				[&](const tbb::blocked_range<size_t> &r) {
					for (size_t l = r.begin(); l < r.end(); ++l)
						loop_island[l] = island[tri_of_loop[l]];
				});
		}
		int64_t n_islands = 1;
		if (T)
			for (size_t t = 0; t < T; ++t)
				n_islands = std::max(n_islands, island[t] + 1);
		R.timer.stage("loop_island");

		// --- group triangles by (material, island) ---
		// one sort of tri_key + split by boundary: members end up in ascending
		// triangle order within each group, groups in ascending key order
		// (same as np.unique(return_inverse) + argsort/nonzero in numpy)
		std::vector<int64_t> tri_key(T);
		if (T) {
			tbb::parallel_for(tbb::blocked_range<size_t>(0, T),
				[&](const tbb::blocked_range<size_t> &r) {
					for (size_t t = r.begin(); t < r.end(); ++t)
						tri_key[t] = (int64_t)mats[t] * n_islands + island[t];
				});
		}
		std::vector<int64_t> g_order = T ? argsort1<int64_t>(tri_key.data(), T) : std::vector<int64_t>();
		std::vector<int64_t> group_start, group_key;
		for (size_t i = 0; i < T; ++i) {
			if (i == 0 || tri_key[g_order[i]] != tri_key[g_order[i - 1]]) {
				group_start.push_back((int64_t)i);
				group_key.push_back(tri_key[g_order[i]]);
			}
		}
		const size_t G = group_start.size();
		R.timer.stage("group");

		// --- bake the transform into points + pre-rotate normals ---
		bool normals_pretransformed = false;
		bool has_mesh_transform = false;
		float mesh_transform[16] = {0};
		if (!is_viewport_render && !use_instancing && has_xform) {
			std::memcpy(mesh_transform, xform, sizeof(xform));
			has_mesh_transform = true;
			float rot[9] = {
				xform[0], xform[1], xform[2],
				xform[4], xform[5], xform[6],
				xform[8], xform[9], xform[10]};
			if (det3(rot) > 0.0) {
				float rinv[9];
				inv3(rot, rinv);
				// points: p' = p @ R^T + t   (p'[j] = sum_k p[k] * R[j][k] + t[j])
				tbb::parallel_for(tbb::blocked_range<size_t>(0, P),
					[&](const tbb::blocked_range<size_t> &r) {
						for (size_t p = r.begin(); p < r.end(); ++p) {
							float *dst = &R.welded_points[p * 3];
							float px = dst[0], py = dst[1], pz = dst[2];
							for (int j = 0; j < 3; ++j)
								dst[j] = px * rot[j * 3] + py * rot[j * 3 + 1] + pz * rot[j * 3 + 2]
									+ xform[j * 4 + 3];
						}
					});
				// normals: n' = n @ Rinv then normalize (n'[j] = sum_k n[k] * Rinv[k][j])
				tbb::parallel_for(tbb::blocked_range<size_t>(0, L),
					[&](const tbb::blocked_range<size_t> &r) {
						for (size_t l = r.begin(); l < r.end(); ++l) {
							float *n = &ln[l * 3];
							float nx = n[0], ny = n[1], nz = n[2];
							float rx = nx * rinv[0] + ny * rinv[3] + nz * rinv[6];
							float ry = nx * rinv[1] + ny * rinv[4] + nz * rinv[7];
							float rz = nx * rinv[2] + ny * rinv[5] + nz * rinv[8];
							float norm = std::sqrt(rx * rx + ry * ry + rz * rz);
							float inv = 1.0f / std::max(norm, 1e-30f);
							n[0] = rx * inv;
							n[1] = ry * inv;
							n[2] = rz * inv;
						}
					});
				has_mesh_transform = false;
				normals_pretransformed = true;
			}
		}
		R.timer.stage("transform");

		// --- per-object shared candidate arrays ---
		// layout: uv layers (k=2 each), normal (k=3), then per color attr:
		// CORNER -> rgb (k=3) + alpha (k=1) candidates; POINT -> welded
		// pt_rgb (k=3) + pt_alpha (k=1)
		// cand[loop_welded[l]] = loop_data[l], last writer wins (loop order)
		auto make_candidate = [&](const float *loop_data, size_t k) {
			std::vector<float> cand(P * k, 0.0f);
			for (size_t l = 0; l < L; ++l)
				std::memcpy(&cand[loop_welded[l] * k], &loop_data[l * k], k * 4);
			return cand;
		};
		for (size_t u = 0; u < n_uv; ++u) {
			R.candidates.push_back(make_candidate(uv_ptrs[u], 2));
			R.candidate_cols.push_back(2);
		}
		R.candidates.push_back(make_candidate(ln, 3));
		R.candidate_cols.push_back(3);
		const int NORMAL_CAND = (int)R.candidates.size() - 1;

		// contiguous corner color arrays (rgb/alpha) per CORNER attr
		std::vector<std::vector<float>> corner_rgb_store(color_attrs.size());
		std::vector<std::vector<float>> corner_alpha_store(color_attrs.size());
		std::vector<const float *> corner_rgb(color_attrs.size(), nullptr);
		std::vector<const float *> corner_alpha(color_attrs.size(), nullptr);
		std::vector<int> color_cand_rgb(color_attrs.size(), -1);
		std::vector<int> color_cand_alpha(color_attrs.size(), -1);
		std::vector<int> point_cand_rgb(color_attrs.size(), -1);
		std::vector<int> point_cand_alpha(color_attrs.size(), -1);
		for (size_t i = 0; i < color_attrs.size(); ++i) {
			if (color_domains[i] == "CORNER") {
				size_t n = col_rows[i];
				corner_rgb_store[i].resize(n * 3);
				corner_alpha_store[i].resize(n);
				for (size_t l = 0; l < n; ++l) {
					corner_rgb_store[i][l * 3] = col_ptrs[i][l * 4];
					corner_rgb_store[i][l * 3 + 1] = col_ptrs[i][l * 4 + 1];
					corner_rgb_store[i][l * 3 + 2] = col_ptrs[i][l * 4 + 2];
					corner_alpha_store[i][l] = col_ptrs[i][l * 4 + 3];
				}
				corner_rgb[i] = corner_rgb_store[i].data();
				corner_alpha[i] = corner_alpha_store[i].data();
				R.candidates.push_back(make_candidate(corner_rgb[i], 3));
				R.candidate_cols.push_back(3);
				color_cand_rgb[i] = (int)R.candidates.size() - 1;
				R.candidates.push_back(make_candidate(corner_alpha[i], 1));
				R.candidate_cols.push_back(1);
				color_cand_alpha[i] = (int)R.candidates.size() - 1;
			} else if (color_domains[i] == "POINT") {
				// per-vertex colors welded once: colors[first_index], split rgb/alpha
				std::vector<float> rgb(P * 3), alpha(P);
				for (size_t p = 0; p < P; ++p) {
					const float *src = &col_ptrs[i][first_index[p] * 4];
					rgb[p * 3] = src[0];
					rgb[p * 3 + 1] = src[1];
					rgb[p * 3 + 2] = src[2];
					alpha[p] = src[3];
				}
				R.candidates.push_back(std::move(rgb));
				R.candidate_cols.push_back(3);
				point_cand_rgb[i] = (int)R.candidates.size() - 1;
				R.candidates.push_back(std::move(alpha));
				R.candidate_cols.push_back(1);
				point_cand_alpha[i] = (int)R.candidates.size() - 1;
			} else {
				throw std::runtime_error("compute_submeshes_c: unhandled color domain '" + color_domains[i] + "'");
			}
		}
		R.timer.stage("candidates");

		// --- per-submesh payloads (parallel over groups) ---
		R.submeshes.resize(G);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, G),
			[&](const tbb::blocked_range<size_t> &gr) {
				for (size_t g = gr.begin(); g < gr.end(); ++g) {
					int64_t gkey = group_key[g];
					uint32_t mat = (uint32_t)(gkey / n_islands);
					uint32_t isl = (uint32_t)(gkey % n_islands);
					int64_t g_start = group_start[g];
					int64_t g_end = (g + 1 < G) ? group_start[g + 1] : (int64_t)T;
					size_t g_n = (size_t)(g_end - g_start);

					SubmeshOut &sm = R.submeshes[g];
					snprintf(sm.name, sizeof(sm.name), "%s%03ui%03u", mesh_key.c_str(), mat, isl);
					sm.mat = mat;
					sm.slot = (int)g + 1;
					sm.normals_pretransformed = normals_pretransformed;

					// Single fused pass over this group's corners: loop ids,
					// welded corner ids and (share mode) the triangle array
					const bool whole_mesh = (g_n == (size_t)T);
					ScalableVector<int64_t> g_loops(g_n * 3);
					ScalableVector<int64_t> gw(g_n * 3);
					if (share_points)
						sm.triangles.resize(g_n * 3);
					for (size_t j = 0; j < g_n; ++j) {
						int64_t t = g_order[g_start + j];
						for (int c = 0; c < 3; ++c) {
							size_t i = j * 3 + c;
							int64_t loop = tl[t * 3 + c];
							int64_t w = loop_welded[loop];
							g_loops[i] = loop;
							gw[i] = w;
							if (share_points)
								sm.triangles[i] = (uint32_t)w;
						}
					}

					if (share_points) {
						sm.n_points = P;
						// Island loops of this submesh. When the group covers the
						// whole mesh there is a single island, so the island filter
						// is a no-op and the corner arrays can be used directly
						ScalableVector<int64_t> isl_loops, isl_gw;
						const int64_t *isl_loops_p;
						const int64_t *isl_gw_p;
						size_t isl_n;
						if (whole_mesh) {
							isl_loops_p = g_loops.data();
							isl_gw_p = gw.data();
							isl_n = g_n * 3;
						} else {
							isl_loops.reserve(g_n * 3);
							isl_gw.reserve(g_n * 3);
							for (size_t i = 0; i < g_n * 3; ++i)
								if (loop_island[g_loops[i]] == (int64_t)isl) {
									isl_loops.push_back(g_loops[i]);
									isl_gw.push_back(gw[i]);
								}
							isl_loops_p = isl_loops.data();
							isl_gw_p = isl_gw.data();
							isl_n = isl_loops.size();
						}
						auto submesh_attr = [&](int cand, const float *loop_data, size_t k) -> AttrRef {
							// share the candidate if every corner agrees with it
							// (memcmp: bitwise row compare, SIMD-friendly; rows with
							// identical NaN payloads compare equal, which is fine since
							// the shared data would be bit-identical anyway)
							if (cand >= 0) {
								const float *c = R.candidates[cand].data();
								bool equal = true;
								for (size_t i = 0; i < g_n * 3; ++i) {
									if (std::memcmp(&loop_data[g_loops[i] * k], &c[gw[i] * k],
											k * sizeof(float)) != 0) {
										equal = false;
										break;
									}
								}
								if (equal)
									return AttrRef{cand, {}};
							}
							AttrRef ref;
							// uninitialized: rows outside the submesh's referenced
							// points are never read (same as np.empty in the
							// numpy fallback)
							ref.owned.reset(static_cast<float *>(
								scalable_malloc(sizeof(float) * P * k)));
							ref.owned_size = P * k;
							for (size_t i = 0; i < isl_n; ++i)
								std::memcpy(&ref.owned[isl_gw_p[i] * k], &loop_data[isl_loops_p[i] * k], k * 4);
							return ref;
						};
						for (size_t u = 0; u < n_uv; ++u)
							sm.uvs.push_back(submesh_attr((int)u, uv_ptrs[u], 2));
						sm.normals = submesh_attr(NORMAL_CAND, ln, 3);
						for (size_t i = 0; i < color_attrs.size(); ++i) {
							if (color_domains[i] == "POINT") {
								sm.colors.push_back(AttrRef{point_cand_rgb[i], {}});
								sm.alphas.push_back(AttrRef{point_cand_alpha[i], {}});
							} else {
								sm.colors.push_back(submesh_attr(color_cand_rgb[i], corner_rgb[i], 3));
								sm.alphas.push_back(submesh_attr(color_cand_alpha[i], corner_alpha[i], 1));
							}
						}
						if (g == 0) {
							sm.points_shared = true;
							sm.has_transform = has_mesh_transform;
							if (has_mesh_transform)
								std::memcpy(sm.transform, mesh_transform, sizeof(sm.transform));
						} else {
							sm.points_is_name = true;
							std::strncpy(sm.points_name, R.submeshes[0].name, sizeof(sm.points_name) - 1);
						}
						// Creases live once in R; only the first submesh references
						// them (BLC only reads creases from submesh 0)
						sm.has_creases = has_edge_verts && (g == 0);
					} else {
						// compact: unique used welded points, local remap
						ScalableVector<int64_t> used(gw.begin(), gw.end());
						std::sort(used.begin(), used.end());
						used.erase(std::unique(used.begin(), used.end()), used.end());
						const size_t n_used = used.size();
						sm.n_points = n_used;
						ScalableVector<int64_t> local_idx(g_n * 3);
						for (size_t i = 0; i < g_n * 3; ++i)
							local_idx[i] = std::lower_bound(used.begin(), used.end(), gw[i]) - used.begin();
						sm.triangles.resize(g_n * 3);
						for (size_t i = 0; i < g_n * 3; ++i)
							sm.triangles[i] = (uint32_t)local_idx[i];
						sm.points_owned.resize(n_used * 3);
						for (size_t i = 0; i < n_used; ++i)
							std::memcpy(&sm.points_owned[i * 3], &R.welded_points[used[i] * 3], 12);
						sm.has_transform = has_mesh_transform;
						if (has_mesh_transform)
							std::memcpy(sm.transform, mesh_transform, sizeof(sm.transform));
						auto compact_attr = [&](const float *loop_data, size_t k) -> AttrRef {
							AttrRef ref;
							ref.owned.reset(static_cast<float *>(
								scalable_malloc(sizeof(float) * n_used * k)));
							ref.owned_size = n_used * k;
							for (size_t i = 0; i < g_n * 3; ++i)
								std::memcpy(&ref.owned[local_idx[i] * k], &loop_data[g_loops[i] * k], k * 4);
							return ref;
						};
						for (size_t u = 0; u < n_uv; ++u)
							sm.uvs.push_back(compact_attr(uv_ptrs[u], 2));
						sm.normals = compact_attr(ln, 3);
						for (size_t i = 0; i < color_attrs.size(); ++i) {
							if (color_domains[i] == "POINT") {
								AttrRef r, a;
								r.owned.reset(static_cast<float *>(
									scalable_malloc(sizeof(float) * n_used * 3)));
								r.owned_size = n_used * 3;
								a.owned.reset(static_cast<float *>(
									scalable_malloc(sizeof(float) * n_used)));
								a.owned_size = n_used;
								const float *prgb = R.candidates[point_cand_rgb[i]].data();
								const float *pal = R.candidates[point_cand_alpha[i]].data();
								for (size_t p = 0; p < n_used; ++p) {
									std::memcpy(&r.owned[p * 3], &prgb[used[p] * 3], 12);
									a.owned[p] = pal[used[p]];
								}
								sm.colors.push_back(std::move(r));
								sm.alphas.push_back(std::move(a));
							} else {
								sm.colors.push_back(compact_attr(corner_rgb[i], 3));
								sm.alphas.push_back(compact_attr(corner_alpha[i], 1));
							}
						}
						// no creases without subdivision
					}
				}
			});
		R.timer.stage("payloads");

		R.duration = std::chrono::duration<double>(Clock::now() - wall0).count();
		Py_END_ALLOW_THREADS
	}

	// ---- wrap results into python objects (GIL held) ----
	std::vector<py::array> cand_arrays(R.candidates.size());
	for (size_t i = 0; i < R.candidates.size(); ++i)
		cand_arrays[i] = wrap_owned<float>(std::move(R.candidates[i]),
			{(py::ssize_t)R.num_points, (py::ssize_t)R.candidate_cols[i]});
	py::array welded_arr = wrap_owned<float>(std::move(R.welded_points),
		{(py::ssize_t)R.num_points, 3});

	auto wrap_attr = [&](AttrRef &ref, size_t P, int k) -> py::array {
		if (ref.cand >= 0)
			return py::reinterpret_borrow<py::array>(cand_arrays[ref.cand]);
		return wrap_owned_buf<float>(std::move(ref.owned), {(py::ssize_t)P, (py::ssize_t)k});
	};

	// Crease arrays are shared by all submeshes: wrap them once and borrow
	py::array crease_pairs_arr, crease_weights_arr;
	if (R.has_creases) {
		py::ssize_t npairs = (py::ssize_t)R.crease_pairs.size();
		py::ssize_t nweights = (py::ssize_t)R.crease_weights.size();
		crease_pairs_arr = wrap_owned<uint32_t>(std::move(R.crease_pairs), {npairs});
		crease_weights_arr = wrap_owned<float>(std::move(R.crease_weights), {nweights});
	}

	py::list payloads;
	for (auto &sm : R.submeshes) {
		py::dict d;
		d["name"] = sm.name;
		if (sm.points_is_name)
			d["points"] = py::str(sm.points_name);
		else if (sm.points_shared)
			d["points"] = welded_arr;
		else
			d["points"] = wrap_owned<float>(std::move(sm.points_owned), {(py::ssize_t)sm.n_points, 3});
		d["triangles"] = wrap_owned<uint32_t>(std::move(sm.triangles),
			{(py::ssize_t)(sm.triangles.size() / 3), 3});
		d["normals"] = wrap_attr(sm.normals, sm.n_points, 3);
		py::list uvs, colors, alphas;
		for (auto &a : sm.uvs) uvs.append(wrap_attr(a, sm.n_points, 2));
		for (auto &a : sm.colors) colors.append(wrap_attr(a, sm.n_points, 3));
		for (auto &a : sm.alphas) alphas.append(wrap_attr(a, sm.n_points, 1));
		d["uvs"] = uvs;
		d["colors"] = colors;
		d["alphas"] = alphas;
		if (sm.has_transform)
			d["transformation"] = wrap_owned<float>(
				std::vector<float>(sm.transform, sm.transform + 16), {4, 4});
		else
			d["transformation"] = py::none();
		if (sm.has_creases) {
			d["crease_pairs"] = py::reinterpret_borrow<py::array>(crease_pairs_arr);
			d["crease_weights"] = py::reinterpret_borrow<py::array>(crease_weights_arr);
		} else {
			d["crease_pairs"] = py::none();
			d["crease_weights"] = py::none();
		}
		d["normals_pretransformed"] = sm.normals_pretransformed;
		d["mat"] = sm.mat;
		d["slot"] = sm.slot;
		payloads.append(d);
	}

	py::dict info;
	info["duration"] = R.duration;
	info["num_points"] = (py::ssize_t)R.num_points;
	info["num_raw"] = (py::ssize_t)R.num_raw;
	py::dict stages;
	for (auto &s : R.timer.stages)
		stages[s.first.c_str()] = s.second;
	info["stages"] = stages;
	// debug/A-B: per-triangle island labels, so the numpy fallback can be
	// compared against this implementation exactly
	{
		py::array_t<int64_t> isl(R.island_debug.size());
		auto m = isl.mutable_unchecked<1>();
		for (size_t i = 0; i < R.island_debug.size(); ++i)
			m(i) = R.island_debug[i];
		info["island"] = isl;
	}

	return py::make_tuple(payloads, info);
}
