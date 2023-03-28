from dataclasses import dataclass

from dacite import from_dict
import yaml

@dataclass
class FramesConfig:
    step: int
    start: int
    end: int


@dataclass
class CapturingConfig:
    type: str
    source: str
    save_results: bool
    save_images: bool
    output_path: str
    frames: FramesConfig

@dataclass
class DrawSettingsConfig:
    markers: bool
    marker_axes: bool
    id_pose_data: bool
    distance: bool
    leds: bool
    lines: bool
    points: bool


@dataclass
class SetupConfig:
    show_image: bool
    draw_settings: DrawSettingsConfig


@dataclass
class ModelConfig:
    capturing: CapturingConfig
    setup: SetupConfig

    @staticmethod
    def from_config(config):
        return from_dict(data_class=ModelConfig, data=config)

    @staticmethod
    def from_yaml_file(path):
        with open(path, "r") as f:
            config_yaml = yaml.load(f, Loader=yaml.UnsafeLoader)
        return from_dict(data_class=ModelConfig, data=config_yaml)
