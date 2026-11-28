# Smuggler’s Cove · Tidal Landing

Caustic Canvas arranges assets from Poly Haven’s [Smuggler’s Cove collection](https://polyhaven.com/collections/smugglers_cove) around a working jetty. The original models and sand texture are [CC0](https://polyhaven.com/license).

- [Coast Land Rocks 02](https://polyhaven.com/a/coast_land_rocks_02) and [Coast Sand 01](https://polyhaven.com/a/coast_sand_01)
- [Modular Wooden Pier](https://polyhaven.com/a/modular_wooden_pier) and [Treasure Chest](https://polyhaven.com/a/treasure_chest)
- [Wooden Crate 01](https://polyhaven.com/a/wooden_crate_01), [Wooden Barrels 01](https://polyhaven.com/a/wooden_barrels_01) and [Wooden Lantern 01](https://polyhaven.com/a/wooden_lantern_01)

Caustic Canvas modeled the stone loading quay, its four tidal steps, the fishing-gear shed, the timber drying rack and its woven net, the mooring bollards and their lines, the rowboat and oars, the dune grasses and the beach. The steps meet the measured deck height of the wooden pier. The quay's limestone and sawn timber use project-authored textures. The skiff can reuse the pier's wood texture. The shallow water uses a project-authored base-color map and small modeled ripples to receive projected light.

The loader prefers Poly Haven's crate, barrels and lanterns. It uses project-modeled slatted crates, coopered barrels and caged lanterns when those secondary downloads are unavailable. If the color texture for the original coastal rock scan is unavailable, the scan can use project-authored rock coloring. The attribution above covers the original Poly Haven assets.

The hand-cranked cargo crane, fish cart and catch, painted pot markers, tidal rock shelf and limpets were modeled for Caustic Canvas. The shallow-water color atlas incorporates Coast Sand 01, aligned across the waterline and tinted with depth. The ripples, pool surfaces and shell clusters are geometry that can receive painted light. Caustic Canvas authored the fine wave normals and the limestone roughness map used by the harbor materials.

Caustic Canvas also modeled the anchored lugger and its rigging, the covered fish market and boardwalk, and the curved breakwater with its pierhead light. The sailcloth and market linen use project-authored textures. The breakwater shares the quay’s stone texture. The expanded tidal atlas uses the same Coast Sand 01 source as the beach.

The project also modeled the fish market’s bent-willow lobster pots, the net loft’s sailcloth canopy, glass floats and oar rack, and the mussels and kelp on the tidal shelf. The pilot landing, rescue ring and windsock furnish both sides of the breakwater. The lugger carries rope fenders and a cedar net roller, and the skiff is secured to the jetty by two bronze cleats.

The beach uses a continuous wetness gradient over the Coast Sand 01 photograph. Along the waterline the sand darkens, its grain becomes less prominent, and its roughness approaches the shallow water. The dressed paving stones, gable vents, sail markings, market balance and scallop display are project-authored details.

Caustic Canvas modeled the square lift net and winch on the west shore, the fish auction hall with overlapping clay pantiles, the bell loft with its open skylight, and the pierhead tower and fog bell. The lugger carries a copper-roofed chart cabin and a stocked anchor. The sea stars and barnacle colonies follow the exposed shelf rock. The clay and sea star finishes use locally authored normal and roughness maps.

The loader chooses authored levels of detail and simplifies large meshes for interactive lighting. Poly Haven’s [full Unreal island demo](https://blog.polyhaven.com/sc_ue5/) uses its own layout.

The bundled reef and chest normals were baked from the original dense scan geometry onto the interactive meshes. The local sand and pier normals estimate relief from their corresponding color photographs. Sand occlusion was estimated from that relief. The chest’s local roughness and metallic mask follows the iron fittings and wooden panels in its color atlas. These files were produced for Caustic Canvas. Timber, stone, fabric, rope, enamel and other project-modeled surfaces use local finish maps, shared across matching materials. The publisher's normal, roughness and metallic maps take precedence when available.

The local harbor records the filenames and publisher checksums for missing maps and optional 2K color upgrades in `preview/source-textures.json`. Room applies verified images as they become available. To save those images in `scene/smugglers-cove/preview/originals`, run `npm run restore:materials` in the project root. This command updates textures on the existing layout. To use an offline copy, set `CAUSTIC_SCENE_SOURCE_DIR` to the directory or ZIP containing the listed images. When a local asset pack contains other resolutions, `npm run prepare:scenes -- smugglers-cove --force` can rebuild the preset using those files.
