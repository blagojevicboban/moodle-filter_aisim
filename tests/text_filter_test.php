<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

namespace filter_aisim;

/**
 * Unit tests for the AI simulation filter.
 *
 * @package    filter_aisim
 * @copyright  2026 Boban Blagojevic
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 * @covers     \filter_aisim\text_filter
 */
#[\PHPUnit\Framework\Attributes\CoversClass(text_filter::class)]
final class text_filter_test extends \advanced_testcase {
    /**
     * Run the filter on a text in a fresh page.
     *
     * @param string $text
     * @return array [filtered text, JavaScript the page requires]
     */
    private function filter_text(string $text): array {
        global $PAGE;
        $this->resetAfterTest();
        $PAGE = new \moodle_page();
        $PAGE->set_url('/');
        $PAGE->set_context(\context_system::instance());
        $filter = new text_filter(\context_system::instance(), []);
        $filtered = $filter->filter($text);
        return [$filtered, $PAGE->requires->get_end_code()];
    }

    /**
     * Text without a simulation is left alone and loads no JavaScript.
     */
    public function test_plain_text(): void {
        [$filtered, $js] = $this->filter_text('<p>No simulation here.</p>');
        $this->assertSame('<p>No simulation here.</p>', $filtered);
        $this->assertStringNotContainsString('filter_aisim/viewer', $js);
    }

    /**
     * Text with a simulation keeps its markup and loads the viewer, which mounts the simulation in the browser.
     */
    public function test_simulation_loads_viewer(): void {
        $text = '<div data-aisim="convolution" data-image="stripes" data-target="sobelx"></div>';
        [$filtered, $js] = $this->filter_text($text);
        $this->assertSame($text, $filtered);
        $this->assertStringContainsString('filter_aisim/viewer', $js);
    }
}
