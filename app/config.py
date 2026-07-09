from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model_flash: str = "gemini-2.5-flash"
    gemini_model_flash_lite: str = "gemini-2.5-flash-lite"
    gemini_default_model: str = "flash"  # "flash" or "flash-lite"
    skins_dir: str = "skins"
    max_upload_size: int = 5 * 1024 * 1024  # 5MB
    max_image_dimension: int = 768  # longest side, before sending to vision API

    model_config = {"env_file": ".env"}


settings = Settings()
