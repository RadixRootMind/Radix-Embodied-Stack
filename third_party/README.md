# Third-party dependencies

## LeRobot

The tested LeRobot 0.4.4-derived source tree is included in `lerobot/` because
the integration relies on local changes for PI0.5, `hm_pi05`, SO101 degree-mode
control, and checkpoint parsing.

## Houmo model zoo

Copy the `xh_model_zoo` directory from the Houmo XH2 V1.4.0 examples package to:

```text
third_party/xh_model_zoo/
  xh_model_zoo/
    xh_llm/
```

It is excluded from Git because no redistributable license was found in the
deployed copy.
