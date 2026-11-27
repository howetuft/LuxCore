# Shapes, submeshes and shape modifiers

## Overview

A **Shape** (`luxrays::Shape`, `include/luxrays/core/shape.h`) is a handle that
groups the submeshes of one object. It owns the shape name and the **canonical
(shared) vertex buffer**; each submesh is an `ExtTriangleMesh` that references
that buffer.

A **submesh** is one `ExtTriangleMesh` of the group. Submeshes of a shape
share the canonical point buffer, so shared edges stay welded (watertight)
across material slots / islands.

A mesh may store only the vertices its triangles actually reference: the
triangle indices are then **local** (`0..countVertices-1`) and a **vertex map**
(`vertexMap[local] = global`) maps them into the Shape's canonical buffer. All
per-vertex data (positions, normals, UVs, colors, alphas, vertex AOVs) is
compacted through the map at construction (`ExtTriangleMesh::CompactVertexData`);
triangle topology and triangle AOVs are not touched. `MakeGlobalView()` rebuilds
a transient global-indexed view of such a submesh for shape-level work.

## Shape modifiers

**Shape modifiers** (`src/slg/shape_modifiers/`, base class
`slg::ShapeModifier`) transform a whole Shape — e.g. subdivision
(`subdiv.h`), tessellation (`tessellate*.h`), displacement
(`displacement.h`), merge-on-distance, simplify, pointiness, AOV shapes...
They run **once on the source shape's common point buffer** and the result is
split back per submesh, so vertex-level modifiers (displacement, merge) stay
welded and shared edges do not crack.

## Rationale: discontinuity handling

Shape-modifying operations are only well-defined where the surface is
continuous. UV seams, hard edges, material boundaries and creases are
**discontinuities**: across them, vertex data (normals, UVs, colors, AOVs) must
not be interpolated or merged. That is why the exporter splits a mesh into
submeshes at discontinuities (varying-data islands) instead of exporting one
big mesh, and why vertex-level shape modifiers operate on the shared canonical
buffer per shape while resplitting per submesh: discontinuous attributes are
carried per submesh and never blended across a seam.
