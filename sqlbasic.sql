-- step1 :Create the Database
CREATE DATABASE CodeAcademyDB;

-- step2 : use the Database
USE CodeAcademyDB;

-- step3: Create the students table
CREATE TABLE Students (
StudentID INT AUTO_INCREMENT PRIMARY KEY,
FirstName VARCHAR(50) NOT NULL,
LastName VARCHAR(50) NOT NULL,
Email VARCHAR(100) UNIQUE NOT NULL,
EnrollmentDate DATE NOT NULL);

-- step 4 verify the schema

-- show us all table
desc Students;

-- create table by using AS command it like create new one by take cols from exsists table
CREATE TABLE MyStudents 
AS (SELECT StudentID,FirstName,LastName,Email
FROM Students);

CREATE TABLE My_Students 
AS (SELECT StudentID,FirstName,LastName,Email
FROM Students);

-- SELECT BY USING *
SELECT * FROM MyStudents;

-- table drop it will delete the table all 
DROP TABLE MyStudents;
-- The TRUNCATE Removes all rows from a table,Releases the storage space used by that table,You cannot roll back 
TRUNCATE TABLE My_Students;

-- Rename table
RENAME TABLE My_Students TO My__Students;

-- using alter for add new cols
ALTER TABLE Students
ADD (ext INT (4));

-- ALTER TABLE BY USING MODIFY IT USED TO INCREASE COLUMN WIDTH
ALTER TABLE Students
MODIFY COLUMN Email VARCHAR (110);


