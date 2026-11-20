"""What each thing in a scene is to RTX's lidar, as semantic labels a camera's frames can be
segmented by: each mesh labelled (taxonomy `TAXONOMY`) with its bound material's
`omni:simready:nonvisual` base, coating and attributes, as RTX reads them."""

from __future__ import annotations

from pathlib import Path

from darkroom import format
from pxr import Sdf, Usd, UsdGeom, UsdShade

TAXONOMY = "lidar"
NV = "omni:simready:nonvisual"


def of(material: Usd.Prim) -> str:
    """A material's non-visual words, as one label: its base, then its coating and attributes
    if any ("clear_glass + visually_transparent"); "" if it has none."""
    base = material.GetAttribute(f"{NV}:base").Get()
    if not base:
        return ""
    coating = material.GetAttribute(f"{NV}:coating").Get() or "none"
    attributes = material.GetAttribute(f"{NV}:attributes").Get() or []
    said = [
        base,
        *([coating] if coating != "none" else []),
        *(a for a in attributes if a != "none"),
    ]
    return " + ".join(said)


def label(layer: Sdf.Layer, scene: str | Path) -> int:
    """Every mesh of `scene` labelled in `layer` with its lidar material (`of`), and `layer`
    marked for its cameras' frames to be segmented by them (`darkroom.format`). Returns how
    many were labelled."""
    stage = Usd.Stage.Open(str(scene))
    made = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        binding = UsdShade.MaterialBindingAPI(prim)
        material, _ = binding.ComputeBoundMaterial()
        if not material:  # bound a subset at a time (a ground of several covers): each its own
            for subset in binding.GetMaterialBindSubsets():
                bound, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
                if bound and (said := of(bound.GetPrim())):
                    made += _labelled(layer, subset.GetPrim(), said + _grey(bound.GetPrim(), said))
            continue
        said = of(material.GetPrim()) if material else ""
        if not said:
            continue
        made += _labelled(layer, prim, said + _grey(material.GetPrim(), said))
    data = dict(layer.customLayerData)
    data[format.SEGMENT_TAXONOMY] = TAXONOMY
    layer.customLayerData = data
    return made


#: The lidar material whose reflectance is its visible grey (RTX's calibration target, read so
#: with the colour setting on): its label says that grey too (" @ 0.20").
GREY = "calibration_lambertian"


def _grey(material: Usd.Prim, said: str) -> str:
    """ " @ <grey>" for a `GREY` material (its diffuse colour's mean), else ""."""
    if not said.startswith(GREY):
        return ""
    for shader in Usd.PrimRange(material):
        colour = shader.GetAttribute("inputs:diffuseColor") or shader.GetAttribute(
            "inputs:diffuse_color_constant"
        )
        if colour and colour.Get() is not None:
            return f" @ {sum(colour.Get()) / 3:.2f}"
    return ""


def _labelled(layer: Sdf.Layer, prim: Usd.Prim, said: str) -> int:
    """`prim` labelled `said` in `layer` (for segmentation by lidar material). Returns 1."""
    spec = Sdf.CreatePrimInLayer(layer, prim.GetPath())
    api = f"SemanticsLabelsAPI:{TAXONOMY}"
    spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[api]))
    attribute = Sdf.AttributeSpec(
        spec, f"semantics:labels:{TAXONOMY}", Sdf.ValueTypeNames.TokenArray, Sdf.VariabilityUniform
    )
    attribute.default = [said]
    return 1
