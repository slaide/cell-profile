from pathlib import Path
import typing as tp

from cell_profile import Experiment, display

# load experiment metadata

plate_barcode="PL12314-HD3"

cellprofiler_output_path=Path(f"/server/data/cellprofiler/results/{plate_barcode}")

db_uri = 'postgresql://username:password@server.local/imagedb'

experiment=Experiment(
    barcode=plate_barcode,
    db_uri=db_uri
)

cellprofiler_pipelines=experiment.retrieve_cellprofiler_pipelines()
display(cellprofiler_pipelines.head(2))

compound_layout_file:tp.Optional[str]="/home/username/myproject/data/metadata_compound_layout.csv"
if compound_layout_file is None:
    compound_layout=experiment.retrieve_compound_layout(source="db")
else:
    compound_layout=experiment.retrieve_compound_layout(source=compound_layout_file)
display(compound_layout.head(2))

plates_metadata=experiment.retrieve_plates_metadata()

# read data files

for plate_meta in plates_metadata:
    plate_meta.read_files(
        cellprofiler_output_path=cellprofiler_output_path,
        cellprofiler_pipelines=cellprofiler_pipelines,
    )

# process data (filter, normalize, etc.)

pre_normalization_clip_method=dict(level_method="sigma",lower_level=-3,upper_level=3,method="remove")
post_normalization_clip_method=dict(level_method="sigma",lower_level=-4,upper_level=4,method="clip")

plates=[
    plate_meta.process(
        compound_layout,

        timeit=True,
        handle_unused_features="remove",
        unused_feature_threshold_std=0.0001,

        pre_normalize_clip_method=pre_normalization_clip_method,
        post_normalize_clip_method=post_normalization_clip_method,
    )
    for plate_meta
    in plates_metadata
]

# plot data
for cpi in plates:
    cpi.plot(method="pca")
