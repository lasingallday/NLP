-- minimum and maximum project_posted_date's are: '2013-01-01' and '2018-05-01'
-- minimum and maximum project_expiration_date's are: '2013-01-04' and '2018-12-31'
-- max (project post date - expiration date) = 49 days
-- Counts for project current status types:
-- Fully Funded = 826,764
-- Live = 41,851
-- Expired = 241,402
-- Counts of Project's posted after Expiration Date = 29

-- Create percentage_time_till_expired column.
SELECT
  p.id,
  (project_expiration_date - project_posted_date),
  ROUND((project_expiration_date - project_posted_date)::NUMERIC/489.0,3) AS percentage_time_till_expired
FROM public.project p
JOIN public.school s ON p.school_id = s.school_id
JOIN public.teacher t ON p.teacher_id = t.teacher_id
WHERE NOT (project_expiration_date - project_posted_date) < 0

-- Counts of distinct teacher_id's = 395,691
-- max (teacher first post date - project post date) = 5,700 days
-- max teacher_project_posted_sequence = 497

-- Create flag_project_proposal_beginner column.
SELECT DISTINCT
	p.teacher_id,
	p.id,
	CASE
		WHEN p.teacher_project_posted_sequence < 3 THEN 1
    ELSE 0
  END AS flag_project_proposal_beginner
FROM public.project p
JOIN public.school s ON p.school_id = s.school_id
JOIN public.teacher t ON p.teacher_id = t.teacher_id
WHERE NOT (p.project_expiration_date - p.project_posted_date) < 0
GROUP BY p.teacher_id, p.id

-- Put all of this into one table.
CREATE TABLE public.transform_tables AS (
SELECT DISTINCT
	p.teacher_id,
	p.id,
	CASE
		WHEN p.teacher_project_posted_sequence < 3 THEN 1
    ELSE 0
  END AS flag_project_proposal_beginner,
  (project_expiration_date - project_posted_date) AS no_days_until_expiration_date,
  ROUND((project_expiration_date - project_posted_date)::NUMERIC/489.0,3) AS percentage_time_till_expired
FROM public.project p
JOIN public.school s ON p.school_id = s.school_id
JOIN public.teacher t ON p.teacher_id = t.teacher_id
WHERE NOT (p.project_expiration_date - p.project_posted_date) < 0
GROUP BY p.teacher_id, p.id
);
