"""The memory drawings of the local LLM page (spec 8.3): the GPU track of the maps, the strip of the model list and
the RAM bar, as descriptors and as markup, on the pinned snapshot of test_site_fit.py."""

import unittest

from launcher.site import draw, fit
from launcher.tests.test_site_fit import snapshot_bench


class SiteDrawTest(unittest.TestCase):
    def setups(self) -> dict[tuple[str, str], fit.Setup]:
        return {s.key: s for s in snapshot_bench().setups}

    def test_tracks(self):
        b, s = snapshot_bench(), self.setups()
        self.assertEqual(draw.track(s["qwen3.6-35b-a3b", "llamacpp"], fit.Machine(16, 64), 16), "##############7/")
        self.assertEqual(draw.track(s["gemma-4-e4b", "llamacpp"], fit.Machine(8, 16), 16), "#####2./")
        self.assertEqual(draw.track(s["qwen3.6-35b-a3b", "llamacpp"], fit.Machine(12, 32), 16), "###########/|+++g")
        self.assertEqual(draw.track(s["gemma-4-12b", "vllm"], fit.Machine(12, 64), 16), "ooooooooooo/")

    def test_strips(self):
        b, s = snapshot_bench(), self.setups()
        self.assertEqual([draw.scale(b, fit.Machine(g, 64)) for g in fit.GPUS], [15, 15, 16, 24, 32])
        self.assertEqual(draw.strip(s["qwen3.6-35b-a3b", "llamacpp"], fit.Machine(16, 64), 16, 16), "##############7/")
        self.assertEqual(draw.strip(s["qwen3.6-35b-a3b", "llamacpp"], fit.Machine(12, 32), 16, 15), "###########@|++g")
        self.assertEqual(draw.strip(s["gemma-4-e4b", "llamacpp"], fit.Machine(12, 32), 16, 15), "#####2...../___")
        self.assertEqual(draw.strip(s["gemma-4-12b", "vllm"], fit.Machine(12, 64), 16, 15), "~###########/___")

    def test_ram_bar(self):
        s = self.setups()
        bar = draw.ram_bar(s["qwen3.6-35b-a3b", "llamacpp"], fit.Machine(16, 64))
        self.assertEqual(bar[0], ("base", "0.00%", "100.00%"))
        self.assertEqual([x for x in bar if x[0] == "tk"][:2], [("tk", "6.25%", None), ("tk", "12.50%", None)])
        self.assertEqual([x for x in bar if x[0] != "tk"][1:], [("desk", "65.63%", "34.38%"), ("peak", "0.00%", "23.28%"),
                                                                ("cap", "23.28%", "7.97%")])
        over = draw.ram_bar(s["gemma-4-e4b", "llamacpp"], fit.Machine(8, 16))
        self.assertEqual(over[-1], ("end", "80.00%", None))

    def test_markup_of_the_descriptors(self):
        self.assertEqual(draw.track_html("#7./|+a"), '<i class="hb on"></i><i class="hb on p7"></i><i class="hb off"></i>'
                         '<i class="hb desk"></i><i class="hend"></i><i class="hb over"></i><i class="hb over g1"></i>')
        self.assertEqual(draw.track_html("o/"), '<i class="hb line"></i><i class="hb desk"></i>')
        self.assertEqual(draw.strip_html("#3./C|+_"), '<span class="strip"><i class="b t10"><i class="f"></i></i>'
                         '<i class="b t3"><i class="f"></i></i><i class="b"></i><i class="b rs"></i>'
                         '<i class="b rs t3"><i class="f"></i></i><i class="tick"></i><i class="b o t10"><i class="f"></i></i>'
                         '<i class="b x"></i></span>')
        self.assertTrue(draw.strip_html("~#/").startswith('<span class="strip k">'))
        self.assertTrue(draw.strip_html("#" * 21).startswith('<span class="strip dense">'))
        self.assertNotIn("style", draw.ram_svg(draw.ram_bar(self.setups()["qwen3.6-35b-a3b", "llamacpp"], fit.Machine(16, 64)), "map"))

    def test_ram_geometry_of_the_map_bar(self):
        self.assertEqual(draw.RAM_GEOMETRY["map"], {"base": (26, 2, 0), "tk": (20, 6, 0), "desk": (0, 26, 0),
                                                    "peak": (0, 26, 2), "cap": (1, 24, 2), "end": (-4, 34, 0)})
        self.assertEqual(draw.RAM_HEIGHT["map"], 28)                      # spec 8.3's 28 px bar

    def test_a_strip_is_dense_above_twenty_slots(self):
        self.assertTrue(draw.strip_html("#" * 20).startswith('<span class="strip">'))
        self.assertTrue(draw.strip_html("#" * 21).startswith('<span class="strip dense">'))

    def test_the_desktop_is_one_block(self):
        self.assertEqual(fit.DESKTOP_GPU_MIB, 1024, "the track and the strip draw the desktop as one 1 GB block (spec 8.3)")
        s = self.setups()                                 # P26: a share on the test card keeps v1's grammar
        self.assertEqual(draw.track(s["gpt-oss-20b", "vllm"], fit.Machine(16, 64), 16), "###############/|d")


if __name__ == "__main__":
    unittest.main()
