DROP TABLE IF EXISTS public.project_need_statement_most_common_words;
CREATE TABLE public.project_need_statement_most_common_words (
  common_word VARCHAR(100),
  counts_common_word INTEGER
);
ALTER TABLE public.project_need_statement_most_common_words ADD PRIMARY KEY (common_word);
