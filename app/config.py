from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    codebuddy_api_key: str = ""
    codebuddy_base_url: str = "https://api.lkeap.cloud.tencent.com/coding/v3"
    codebuddy_vision_model: str = "hunyuan-2.0-instruct"
    codebuddy_text_model: str = "hunyuan-2.0-instruct"
    skins_dir: str = "skins"
    max_upload_size: int = 5 * 1024 * 1024  # 5MB

    model_config = {"env_file": ".env"}


settings = Settings()
