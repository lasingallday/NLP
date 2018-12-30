-- Write SQL to transform Project, School, and Teacher tables to produce new report per project, teacher, and school.
--Load report here--
DROP TABLE IF EXISTS public.report_all_projects;
CREATE TABLE public.report_all_projects AS (
SELECT DISTINCT
    p.project_id,
    p.teacher_id,
    p.school_id,
    p.project_need_statement,
    p.project_subject_category,
    p.project_subject_subcategory,
    p.project_grade_level_category,
    p.project_resource_category,
    p.project_posted_date,
    p.project_fully_funded_date
FROM public.project p
JOIN public.school s ON p.school_id = s.school_id
JOIN public.teacher t ON p.teacher_id = t.teacher_id
);
-- Next, if it is useful, add more project columns.

-- Statistics:
-- Number distinct project_id's = 1,110,015
-- Number of projects w/o teacher_id in public.teacher = 9
-- Number of projects w/o school_id in public.school = 14
DROP TABLE IF EXISTS public.no_teacher_id;
CREATE TABLE public.no_teacher_id AS (
WITH stuff AS (
SELECT DISTINCT
    p.project_id,
    p.id
FROM public.project p
JOIN public.teacher t ON p.teacher_id = t.teacher_id
)
SELECT p.id, t.teacher_id, t.teacher_prefix, t.teacher_first_project_posted_date
FROM public.project p
LEFT JOIN public.teacher t ON p.teacher_id = t.teacher_id
WHERE NOT EXISTS (SELECT 1 FROM stuff WHERE p.id = stuff.id)
);


--Add flags for 50 most common words
ALTER TABLE public.report_all_projects ADD COLUMN flag_books BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_classroom BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_help BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_learning BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_reading BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_math BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_skills BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_ipad BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_use BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_learn BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_materials BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_class BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_technology BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_new BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_science BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_supplies BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_work BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_school BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_set BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_order BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_center BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_read BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_paper BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_create BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_book BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_chromebooks BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_seating BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_activities BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_literacy BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_access BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_ipads BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_games BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_2 BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_chairs BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_headphones BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_time BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_writing BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_library BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_art BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_practice BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_markers BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_improve BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_two BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_make BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_projects BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_sets BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_enhance BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_also BOOLEAN;
ALTER TABLE public.report_all_projects ADD COLUMN flag_keep BOOLEAN;
