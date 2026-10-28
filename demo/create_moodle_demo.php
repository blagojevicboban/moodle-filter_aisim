<?php
/**
 * CLI script to create an interactive demo course in Moodle 4.5.
 *
 * Usage in WSL:
 *   php create_moodle_demo.php
 */

define('CLI_SCRIPT', true);
require('/home/tsp/moodle45/config.php');
require_once($CFG->dirroot . '/course/lib.php');
require_once($CFG->dirroot . '/course/modlib.php');
require_once($CFG->libdir . '/filterlib.php');

global $DB, $CFG;

echo "=== Kreiranje AI for VET Demo kursa u Moodle 4.5 ===" . PHP_EOL;

// 1. Ensure filter_aisim is active.
filter_set_global_state('aisim', TEXTFILTER_ON);
echo "[OK] Filter filter_aisim je uključen (TEXTFILTER_ON)." . PHP_EOL;

// 2. Find or create Course.
$shortname = 'AIVET_SIM';
$course = $DB->get_record('course', ['shortname' => $shortname]);

if (!$course) {
    $category = $DB->get_record('course_categories', ['id' => 1]);
    $courseData = new stdClass();
    $courseData->category = $category->id;
    $courseData->fullname = 'AI for VET: Interaktivne simulacije (Demo)';
    $courseData->shortname = $shortname;
    $courseData->summary = 'Zvanični demo kurs za projekat AI for VET sa interaktivnim simulacijama veštačke inteligencije i mašinskog učenja.';
    $courseData->summaryformat = FORMAT_HTML;
    $courseData->format = 'topics';
    $courseData->numsections = 4;
    $courseData->visible = 1;

    $course = create_course($courseData);
    echo "[OK] Kreiran nov Moodle kurs: ID={$course->id} ('{$course->fullname}')." . PHP_EOL;
} else {
    echo "[INFO] Kurs već postoji: ID={$course->id} ('{$course->fullname}')." . PHP_EOL;
}

// 3. Helper to create Page activity.
function add_demo_page($course, $sectionNum, $name, $htmlContent) {
    global $DB, $CFG;

    // Check if page already exists in this course.
    $existing = $DB->get_record('page', ['course' => $course->id, 'name' => $name]);
    if ($existing) {
        $existing->content = $htmlContent;
        $DB->update_record('page', $existing);
        echo "  [UPDATE] Stranica '{$name}' je ažurirana." . PHP_EOL;
        return;
    }

    $page = new stdClass();
    $page->course = $course->id;
    $page->name = $name;
    $page->intro = '<p>Interaktivna lekcija sa simulacijom.</p>';
    $page->introformat = FORMAT_HTML;
    $page->content = $htmlContent;
    $page->contentformat = FORMAT_HTML;
    $page->timemodified = time();

    $pageId = $DB->insert_record('page', $page);

    // Create course module.
    $mod = $DB->get_record('modules', ['name' => 'page']);
    $cm = new stdClass();
    $cm->course = $course->id;
    $cm->module = $mod->id;
    $cm->instance = $pageId;
    $cm->section = $sectionNum;
    $cm->visible = 1;
    $cm->visibleold = 1;
    $cm->groupmode = 0;
    $cm->groupingid = 0;
    $cm->completion = 0;
    $cmId = add_course_module($cm);

    $section = $DB->get_record('course_sections', ['course' => $course->id, 'section' => $sectionNum]);
    if (!$section) {
        $section = new stdClass();
        $section->course = $course->id;
        $section->section = $sectionNum;
        $section->summary = '';
        $section->summaryformat = FORMAT_HTML;
        $section->visible = 1;
        $section->sequence = '';
        $sectionId = $DB->insert_record('course_sections', $section);
        $section->id = $sectionId;
    }

    course_add_cm_to_section($course, $cmId, $sectionNum);
    echo "  [OK] Kreirana nova aktivnost '{$name}' (cmid={$cmId})." . PHP_EOL;
}

// 4. Extract content bodies from our demo HTML files.
$dir = __DIR__;

$p1 = file_get_contents($dir . '/01_gradijentni_spust.html');
preg_match('/<div class="page-wrap">(.*?)<div class="nav-links">/s', $p1, $m1);
$content1 = $m1 ? $m1[1] : '<div data-aisim="gradient-descent"></div>';

$p2 = file_get_contents($dir . '/02_matrica_konfuzije.html');
preg_match('/<div class="page-wrap">(.*?)<div class="nav-links">/s', $p2, $m2);
$content2 = $m2 ? $m2[1] : '<div data-aisim="confusion-matrix"></div>';

$p3 = file_get_contents($dir . '/03_polinomska_regresija.html');
preg_match('/<div class="page-wrap">(.*?)<div class="nav-links">/s', $p3, $m3);
$content3 = $m3 ? $m3[1] : '<div data-aisim="regression"></div>';

$p4 = file_get_contents($dir . '/04_granice_odlucivanja.html');
preg_match('/<div class="page-wrap">(.*?)<div class="nav-links">/s', $p4, $m4);
$content4 = $m4 ? $m4[1] : '<div data-aisim="knn-tree"></div>';

// 5. Add all 4 Pages to course.
add_demo_page($course, 1, '1. Optimizacija: Gradijentni spust (Gradient Descent)', $content1);
add_demo_page($course, 2, '2. Evaluacija: Matrica konfuzije i prag odlučivanja', $content2);
add_demo_page($course, 3, '3. Generalizacija: Polinomska regresija i preprilagođavanje', $content3);
add_demo_page($course, 4, '4. Klasifikacija: Granice odlučivanja (k-NN i Stablo)', $content4);

rebuild_course_cache($course->id, true);

// Enable guest access.
require_once($CFG->dirroot . '/enrol/guest/lib.php');
$guestPlugin = enrol_get_plugin('guest');
$guestInstance = $DB->get_record('enrol', ['courseid' => $course->id, 'enrol' => 'guest']);
if (!$guestInstance) {
    $guestPlugin->add_instance($course, ['status' => ENROL_INSTANCE_ENABLED]);
} else {
    $DB->set_field('enrol', 'status', ENROL_INSTANCE_ENABLED, ['id' => $guestInstance->id]);
}

purge_all_caches();

echo PHP_EOL . "=== Uspešno kreirano! ===" . PHP_EOL;
echo "Direktan link ka kursu na vašem lokalnom serveru:" . PHP_EOL;
echo "👉 http://localhost/m45/course/view.php?id={$course->id}" . PHP_EOL;
