from pydantic import BaseModel, validator


class EmbeddingType(BaseModel):
    type: str

    @validator("type")
    def check_action_type(cls, value):
        valid_types = ["huggingface", "openai"]
        if value not in valid_types:
            raise ValueError(
                "{} is invalid. homography type can only be {}".format(
                    value, " or ".join(valid_types)
                )
            )
        return value


class LLMType(BaseModel):
    type: str

    @validator("type")
    def check_action_type(cls, value):
        valid_types = ["huggingface", "openai"]
        if value not in valid_types:
            raise ValueError(
                "{} is invalid. homography type can only be {}".format(
                    value, " or ".join(valid_types)
                )
            )
        return value
