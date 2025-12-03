from enum import Enum


class RerankMethod(Enum):
    INDIVIDUAL = "individual"
    BATCH = "batch"
    CROSS_ENCODER = "cross_encoder"
