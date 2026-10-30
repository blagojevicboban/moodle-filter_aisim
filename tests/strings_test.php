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
 * The simulation engines and the language files agree on the string identifiers.
 *
 * The engines get their texts from the language files at run time, so a missing string only shows up as [[key]]
 * in the browser. These tests catch it earlier.
 *
 * @package    filter_aisim
 * @copyright  2026 Boban Blagojevic
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 * @coversNothing
 */
final class strings_test extends \basic_testcase {
    /**
     * Source of every simulation engine.
     *
     * @return array file name => JavaScript source
     */
    private function engines(): array {
        $sources = [];
        foreach (glob(__DIR__ . '/../amd/src/sims/*.js') as $file) {
            $sources[basename($file)] = file_get_contents($file);
        }
        $this->assertNotEmpty($sources);
        return $sources;
    }

    /**
     * Strings of a language file in the plugin.
     *
     * @param string $lang
     * @return array identifier => text
     */
    private function lang_strings(string $lang): array {
        $string = [];
        include(__DIR__ . "/../lang/{$lang}/filter_aisim.php");
        return $string;
    }

    /**
     * Each engine lists exactly the strings it uses, and the English language file has all of them.
     */
    public function test_engines_use_listed_strings(): void {
        $en = $this->lang_strings('en');
        foreach ($this->engines() as $file => $source) {
            preg_match_all("/this\\.str\\('([a-z0-9_]+)'/", $source, $used);
            $this->assertMatchesRegularExpression('/\\.STRINGS = \\[([^\\]]*)\\]/', $source, $file);
            preg_match('/\\.STRINGS = \\[([^\\]]*)\\]/', $source, $list);
            preg_match_all("/'([a-z0-9_]+)'/", $list[1], $listed);

            $used = array_unique($used[1]);
            sort($used);
            $listed = $listed[1];
            sort($listed);
            $this->assertSame($used, $listed, "$file: STRINGS must list exactly the keys passed to this.str()");
            foreach ($used as $key) {
                $this->assertArrayHasKey($key, $en, "$file uses '$key', which lang/en does not define");
            }
        }
    }

    /**
     * The bundled translations have the same strings and placeholders as English.
     */
    public function test_translations_complete(): void {
        $en = $this->lang_strings('en');
        foreach (glob(__DIR__ . '/../lang/*', GLOB_ONLYDIR) as $dir) {
            $lang = basename($dir);
            if ($lang === 'en') {
                continue;
            }
            $strings = $this->lang_strings($lang);
            $this->assertEqualsCanonicalizing(array_keys($en), array_keys($strings), "lang/$lang keys");
            foreach ($en as $key => $text) {
                preg_match_all('/\{\$a[^}]*\}/', $text, $expected);
                preg_match_all('/\{\$a[^}]*\}/', $strings[$key], $actual);
                $this->assertEqualsCanonicalizing($expected[0], $actual[0], "lang/$lang '$key' placeholders");
            }
        }
    }
}
