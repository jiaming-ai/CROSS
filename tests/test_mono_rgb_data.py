import cv2
import numpy as np
import pytest

from cross.mono.data import RGBSequence


def color_package(path):
    (path/"color").mkdir()
    image = np.arange(8*12*3, dtype=np.uint8).reshape(8, 12, 3)
    cv2.imwrite(str(path/"color/first.png"), image)
    (path/"color.txt").write_text("1.25 color/first.png\n")
    handle = cv2.FileStorage(str(path/"sensors.yaml"), cv2.FILE_STORAGE_WRITE)
    try:
        handle.startWriteStruct("d400_color_optical_frame", cv2.FileNode_MAP)
        handle.write("model", "pinhole")
        handle.write("width", 12)
        handle.write("height", 8)
        handle.write("distortion_model", "radial-tangential")
        handle.write("intrinsics", np.array([[10., 5., 11., 4.]]))  # fx,cx,fy,cy
        handle.write("distortion_coefficients", np.zeros((1, 5)))
        handle.endWriteStruct()
    finally:
        handle.release()
    return image


def test_openloris_reader_needs_only_color_and_calibration(tmp_path):
    image = color_package(tmp_path)
    sequence = RGBSequence(tmp_path)
    # No depth, odometry, extrinsics, IMU or ground truth exists in this fixture.
    np.testing.assert_array_equal(sequence.K, [[10., 0, 5.], [0, 11., 4.], [0, 0, 1.]])
    frame = next(iter(sequence))
    np.testing.assert_array_equal(frame.rgb, image[:, :, ::-1])
    assert frame.timestamp == 1.25 and frame.index == 0


def test_resized_intrinsics_preserve_projected_rays_and_pixels(tmp_path):
    image = color_package(tmp_path)
    sequence = RGBSequence(tmp_path, resize=(6, 6))
    point = np.array([.2, -.1, 1.])
    before = sequence.input_K @ point
    expected = (before[:2] + .5) * [.5, .75] - .5
    np.testing.assert_allclose((sequence.K @ point)[:2], expected)
    expected_image = cv2.resize(image, (6, 6), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    frame = next(iter(sequence))
    np.testing.assert_array_equal(frame.rgb, expected_image)
    assert frame.input_timing["resize_wall_seconds"] >= 0
    with pytest.raises(ValueError, match="positive integers"):
        RGBSequence(tmp_path, resize=(0, 6))


def test_input_dimensions_must_match_calibration(tmp_path):
    color_package(tmp_path)
    cv2.imwrite(str(tmp_path/"color/first.png"), np.zeros((4, 12, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="dimensions"):
        next(iter(RGBSequence(tmp_path)))
