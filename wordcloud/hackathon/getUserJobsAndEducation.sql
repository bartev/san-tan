select * from users limit 10;
select * from users where last_name = 'Vartanian' limit 10;
select * from users where first_name = 'Bartev' limit 10;

--id        first_name last_name email                 phone        avatar_file_name     avatar_content_type gender ethnicity_id birthdate  website                                                candidate_terms_of_service_id vanity_id          created_at          updated_at          active lat        lng          fake   avatar_file_size deactivated requires_login 
----------- ---------- --------- --------------------- ------------ -------------------- ------------------- ------ ------------ ---------- ------------------------------------------------------ ----------------------------- ------------------ ------------------- ------------------- ------ ---------- ------------ ------ ---------------- ----------- -------------- 
--211864909 Bartev     Bnza                            (null)       (null)               (null)              male   (null)       1990-06-06 http://www.facebook.com/bartev.bnza                    (null)                        (null)             2012-02-17 16:09:57 2012-10-30 13:49:29 false  (null)     (null)       (null) (null)           (null)      (null)         
--423751524 Bartev     Levon                           (null)       (null)               (null)              male   (null)       1992-07-23 http://www.facebook.com/profile.php?id=100003650824574 (null)                        (null)             2012-08-17 00:56:42 2012-08-17 00:56:42 false  19.4333    -99.1333     (null) (null)           (null)      (null)         
--540890438 Bartev     Karaman                         (null)       (null)               (null)              male   (null)       1992-09-11 http://www.facebook.com/bartev.karaman                 (null)                        (null)             2012-09-05 17:27:07 2012-09-05 17:27:07 false  (null)     (null)       (null) (null)           (null)      (null)         
--83072059  Bartev     Garyan                          (null)       (null)               (null)              (null) (null)       1960-02-10 http://www.facebook.com/bartev.garyan                  (null)                        (null)             2012-09-14 20:00:20 2012-11-30 18:12:04 false  (null)     (null)       (null) (null)           (null)      (null)         

--2803522   Bartev     Vartanian bartev@gmail.com      (null)       avatar1331838474.jpg text/plain          Male   (null)       (null)     (null)                                                 (null)                        bartev             2012-03-15 19:01:22 2013-02-08 19:09:37 true   37.7624642 -121.9814354 (null) 1932             (null)      false          

--615092249 Bartev     Vartanian (null)                111-111-1111 (null)               (null)              (null) (null)       (null)     (null)                                                 (null)                        (null)             2012-06-07 00:34:52 2013-03-06 14:46:49 (null) (null)     (null)       (null) (null)           (null)      (null)         
--5993620   Bartev     Vartanian bartev@identified.com (null)       (null)               (null)              Male   (null)       (null)     (null)                                                 (null)                        bartev-vartanian-1 2012-04-09 20:52:36 2012-04-09 20:55:00 true   (null)     (null)       (null) (null)           (null)      (null)         
--32053895  Bartev     Sakadjian                       (null)       (null)               (null)              male   (null)       0001-07-03 http://www.facebook.com/profile.php?id=589036801       (null)                        (null)             2011-11-02 05:20:10 2011-11-02 05:20:10 false  (null)     (null)       (null) (null)           (null)      (null)         

select * from user_jobs where user_id = 2803522;
select * from ic_iic_codes limit 4;
select * from orgs limit 5;

-- get job history
select j.display_company_name, j.display_job_title_name, o.org_name, i.iic_name, j.description, j.start_date, j.company_summary_id, o.industry_id
	from user_jobs j 
		left outer join orgs o on j.company_summary_id = o.id
		left outer join ic_iic_codes i on o.industry_id = i.iic
	where user_id = 2803522 
	order by start_date desc;

select * from user_education e  limit 4;
select * from public.cu_majors;

-- get education history
select e.display_university_name, e.display_degree_name, e.display_major_name, e.display_minor_name, e.major_id, e.minor_id, e.degree_id, o.org_name, m.major, mm.major as minor, e.start_date, e.end_date
	from user_education e left outer join orgs o on e.university_id = o.id
		left outer join cu_majors m on e.major_id  = m.id
		left outer join cu_majors mm on e.minor_id = mm.id
	where user_id = 2803522
	order by e.start_date desc;
	
-- get friends' id
select * from user_friends limit 4;
select * from user_friends f where f.user_id = 2803522;
	
