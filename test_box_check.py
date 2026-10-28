from PIL import Image, ImageDraw

import box_check as bc

W, H = 1000, 800


def test_orientation_rules():
    assert bc.orientation((100, 400), (600, 400)) == "horizontal"
    assert bc.orientation((100, 400), (600, 450)) == "horizontal"
    assert bc.orientation((500, 100), (500, 700)) == "vertical"
    assert bc.orientation((500, 100), (530, 700)) == "vertical"
    assert bc.orientation((100, 100), (600, 600)) == "diagonal"


def test_box_sizes():
    b = bc.arrow_box((100, 400), (600, 400), W, H)
    assert (b.half_len, b.half_wid) == (250 + 30, 0 + 24)
    b = bc.arrow_box((500, 100), (500, 700), W, H)
    assert (b.half_len, b.half_wid) == (300 + 24, 0 + 30)
    b = bc.arrow_box((100, 100), (500, 400), W, H)
    assert b.axis == (1.0, 0.0)
    assert (b.half_len, b.half_wid) == (200 + 30, 150 + 24)


def test_fraction_inside_threshold():
    gt = bc.arrow_box((100, 400), (600, 400), W, H)
    assert bc.fraction_inside((100, 400), (600, 400), gt) == 1.0
    assert bc.fraction_inside((100, 415), (600, 415), gt) == 1.0
    assert bc.fraction_inside((100, 440), (600, 440), gt) == 0.0
    assert bc.fraction_inside((100, 400), (600, 300), gt) < 0.75
    assert bc.fraction_inside((100, 400), (1100, 400), gt) < 0.75


def test_diagonal_box_is_axis_aligned():
    gt = bc.arrow_box((100, 100), (500, 400), W, H)
    corners = sorted(gt.corners())
    assert corners == [(70, 76), (70, 424), (530, 76), (530, 424)]
    assert bc.fraction_inside((105, 98), (505, 396), gt) == 1.0
    assert bc.fraction_inside((100, 400), (500, 100), gt) == 1.0
    assert bc.fraction_inside((300, 250), (900, 700), gt) < 0.75


def test_box_match_and_direction():
    gt = [((100, 400), (600, 400)), ((500, 100), (500, 300))]
    preds = [
        {"tail": [110, 405], "head": [590, 398]},
        {"tail": [500, 290], "head": [500, 110]},
        {"tail": [800, 700], "head": [900, 700]},
    ]
    r = bc.box_match(gt, preds, W, H)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 0)
    assert sorted(m["direction_ok"] for m in r["matches"]) == [False, True]


def test_margin_schedule():
    assert bc.margin_for_epoch(0, 5, 0.06, 0.03) == 0.06
    assert bc.margin_for_epoch(4, 5, 0.06, 0.03) == 0.03
    assert bc.margin_for_epoch(2, 5) == bc.MARGIN_FRAC


def canvas():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([50, 350, 150, 450], outline="black", width=3)
    d.line([(150, 400), (600, 400)], fill="black", width=3)
    d.line([(700, 100), (700, 300), (900, 300)], fill="black", width=3)
    for x in range(200, 600, 20):
        d.line([(x, 500 + (x - 200) * 0.5), (x + 12, 500 + (x + 12 - 200) * 0.5)], fill="black", width=3)
    return img


def test_pixel_support():
    mask = bc.foreground_mask(canvas())
    on_line = bc.pixel_support(mask, (150, 400), (600, 400))
    assert on_line["continuity"] == 1.0
    empty = bc.pixel_support(mask, (200, 150), (600, 150))
    assert empty["continuity"] == 0.0
    half = bc.pixel_support(mask, (400, 400), (850, 400))
    assert 0.3 < half["continuity"] < 0.75
    elbow = bc.pixel_support(mask, (700, 100), (900, 300))
    assert elbow["path"] == "elbow_vh" and elbow["continuity"] == 1.0
    dashed = bc.pixel_support(mask, (200, 500), (600, 700))
    assert dashed["continuity"] > 0.9


def test_filter_arrows():
    arrows = [
        {"tail": [150, 400], "head": [600, 400]},
        {"tail": [700, 100], "head": [900, 300]},
        {"tail": [200, 150], "head": [600, 150]},
    ]
    kept, report = bc.filter_arrows(canvas(), arrows)
    assert kept == arrows[:2]
    assert [r["kept"] for r in report] == [True, True, False]


def test_choose_path():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([600, 500, 760, 600], fill="#BBBBBB")
    d.line([(680, 500), (680, 200), (300, 200)], fill="black", width=3)
    d.line([(100, 700), (400, 750)], fill="black", width=3)
    mask = bc.stroke_mask(bc.foreground_mask(img))
    name, points = bc.choose_path(mask, (680, 560), (300, 200))
    assert name == "elbow_vh" and points[1] == (680, 200)
    assert bc.choose_path(mask, (100, 100), (400, 400))[0] == "straight"
    assert bc.choose_path(mask, (100, 700), (400, 750)) == ("straight", [(100, 700), (400, 750)])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
