SELECT * FROM ROSSTAT_SALARY_RU
WHERE region_name <> 'Томская область' AND region_name <> 'Самарская область'
ORDER BY salary;

SELECT ROUND(AVG(salary), 2) FROM ROSSTAT_SALARY_RU
WHERE region_name <> 'Томская область' AND region_name <> 'Самарская область';

SELECT percentile_disc(0.5) within group (order by salary)
from ROSSTAT_SALARY_RU
WHERE region_name <> 'Томская область' AND region_name <> 'Самарская область';
