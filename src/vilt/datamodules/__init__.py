from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .yfcc_datamodule import YfccDataModule
from .cc_datamodule import CcDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .msrvtt_datamodule import MSRVTTDataModule
from .webvid_datamodule import WebVIDDataModule
from .imagenet_datamodule import ImageNetDataModule
from .bookcorpus_datamodule import BookCorpusDataModule
from .wikipedia_datamodule import WikipediaDataModule
from .combine_tsv_datamodule import CCSVWDataModule, CCSVDataModule
from .imagenet1k_datamodule import ImageNet1kDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "yfcc": YfccDataModule,
    "cc": CcDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "msrvtt": MSRVTTDataModule,
    "webvid": WebVIDDataModule,
    "imagenet": ImageNetDataModule,
    "bookcorpus": BookCorpusDataModule,
    "wikipedia": WikipediaDataModule,
    "ccsvw": CCSVWDataModule,
    "ccsv": CCSVDataModule,
    "imagenet1k": ImageNet1kDataModule,
}
