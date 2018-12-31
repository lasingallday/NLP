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
-- Add project column for identifying funded projects.
ALTER TABLE public.report_all_projects DROP COLUMN flag_project_funded;
ALTER TABLE public.report_all_projects ADD COLUMN flag_project_funded INTEGER;

UPDATE public.report_all_projects
SET flag_project_funded = 0
WHERE project_fully_funded_date IS NULL;

UPDATE public.report_all_projects
SET flag_project_funded = 1
WHERE project_fully_funded_date IS NOT NULL;


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
