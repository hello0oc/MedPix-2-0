from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "medgemma_gui" / "app.py"


def _build_streamlit_stub() -> types.ModuleType:
    module = types.ModuleType("streamlit")

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Secrets(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    module.secrets = _Secrets()
    module.session_state = {}
    module.sidebar = _Ctx()

    def cache_data(show_spinner=False):
        def _decorator(fn):
            return fn

        return _decorator

    module.cache_data = cache_data
    module.spinner = lambda *_args, **_kwargs: _Ctx()
    module.columns = lambda spec: [_Ctx() for _ in spec]
    module.radio = lambda _label, options, **_kwargs: options[0]
    module.checkbox = lambda _label, value=False, **_kwargs: value
    module.slider = lambda _label, _min, _max, value, _step=1, **_kwargs: value
    module.number_input = lambda _label, _min, _max, value, _step=1, **_kwargs: value
    module.button = lambda *_args, **_kwargs: False
    module.selectbox = lambda _label, options, **_kwargs: options[0]
    module.expander = lambda *_args, **_kwargs: _Ctx()
    module.set_page_config = lambda **_kwargs: None
    module.markdown = lambda *_args, **_kwargs: None
    module.caption = lambda *_args, **_kwargs: None
    module.subheader = lambda *_args, **_kwargs: None
    module.info = lambda *_args, **_kwargs: None
    module.error = lambda *_args, **_kwargs: None
    module.warning = lambda *_args, **_kwargs: None
    module.write = lambda *_args, **_kwargs: None
    module.code = lambda *_args, **_kwargs: None
    module.image = lambda *_args, **_kwargs: None
    module.divider = lambda: None
    module.stop = lambda: None
    return module


@pytest.fixture(scope="session")
def app_module():
    sys.modules["streamlit"] = _build_streamlit_stub()
    spec = importlib.util.spec_from_file_location("medgui_app_test", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def st_stub():
    return sys.modules["streamlit"]
