-- Remove '' and 'NaN' values in project_short_description.
-- (Problems on line 9(project_fully_funded_date) and 3624 (project_short_description))
DROP TABLE IF EXISTS public.project;
CREATE TABLE public.project (
  project_id BYTEA,
  school_id BYTEA,
  teacher_id BYTEA,
  --teacher_project_posted_sequence INTEGER,
  --project_type VARCHAR(100),
  --project_title text,
  project_essay text,
  project_short_description text,
  project_need_statement text,
  project_subject_category VARCHAR(250),
  project_subject_subcategory VARCHAR(250),
  project_grade_level_category VARCHAR(250),
  project_resource_category VARCHAR(250),
  project_cost NUMERIC,
  project_posted_date DATE,
  --project_expiration_date DATE,
  project_current_status VARCHAR(50),
  project_fully_funded_date DATE
);

DROP TABLE IF EXISTS public.school;
CREATE TABLE public.school (
  school_id BYTEA,
  --school_name VARCHAR(250),
  school_morphology VARCHAR(100),
  school_percentage_reduced_free_lunch INTEGER,
  school_state VARCHAR(50),
  --school_zip CHAR(5),
  school_city VARCHAR(100),
  --school_county VARCHAR(100),
  school_district VARCHAR(250)
);

DROP TABLE IF EXISTS public.teacher;
CREATE TABLE public.teacher (
  teacher_id BYTEA,
  teacher_prefix VARCHAR(25),
  teacher_first_project_posted_date DATE
);
