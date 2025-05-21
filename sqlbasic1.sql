USE CodeAcademyDB;

-- one to one relation
CREATE TABLE Employees (
EmployeeID INT PRIMARY KEY,
EmployeeName VARCHAR(50)
);

CREATE TABLE Devices (
DeviceID INT PRIMARY KEY,
EmployeeID INT UNIQUE,
FOREIGN KEY (EmployeeID) REFERENCES Employees(EmployeeID)
);

-- ONE TO MANY RELATIONSHIP(HERE WE PUT THE PK OF ONE AS FK IN MANY)
CREATE TABLE Authors (
AuthorID INT PRIMARY KEY
);

CREATE TABLE Books (
BookID INT PRIMARY KEY,
AuthorID INT,
FOREIGN KEY (AuthorID) REFERENCES Authors(AuthorID)
);

-- MANY TO MANY RELATIONSHIP
CREATE DATABASE UniversityDB;
USE UniversityDB;

CREATE TABLE Students (
StudentID INT PRIMARY KEY
);

CREATE TABLE Courses (
CourseID INT PRIMARY KEY
);

CREATE TABLE StudentCourse (
StudentID INT,
CourseID INT,
PRIMARY KEY (StudentID,CourseID),
FOREIGN KEY (StudentID) REFERENCES Students(StudentID),
FOREIGN KEY (CourseID) REFERENCES Courses(CourseID)
);

-- constaints type

-- table level constraints
CREATE TABLE StudentsTable (
s_id INT,
s_first_name VARCHAR(30),
CONSTRAINT s_studentid_pk PRIMARY KEY (s_id)
);

-- column level constraints in this way we can not give it name in the mysql environment
CREATE TABLE Students_Table (
  s_id INT PRIMARY KEY,
  s_first_name VARCHAR(30)
);

-- Creating constraint when table already created
ALTER TABLE Students_Table
MODIFY COLUMN s_id INT NOT NULL;

-- Add the primary key
ALTER TABLE Students_Table
ADD CONSTRAINT s_studentid_pk PRIMARY KEY (s_id);

-- three different foreign key actions
CREATE TABLE Departments (
  DeptID INT PRIMARY KEY,
  DeptName VARCHAR(50)
);

CREATE TABLE Employees_Cascade_Delete (
  EmpID INT PRIMARY KEY,
  EmpName VARCHAR(50),
  DeptID INT,
  FOREIGN KEY (DeptID) REFERENCES Departments(DeptID)
  ON DELETE CASCADE
);

CREATE TABLE Employees_Cascade_Update (
  EmpID INT PRIMARY KEY,
  EmpName VARCHAR(50),
  DeptID INT,
  FOREIGN KEY (DeptID) REFERENCES Departments(DeptID)
  ON UPDATE CASCADE
);

CREATE TABLE Employees_Set_Null (
  EmpID INT PRIMARY KEY,
  EmpName VARCHAR(50),
  DeptID INT,
  FOREIGN KEY (DeptID) REFERENCES Departments(DeptID)
  ON DELETE SET NULL
);

-- Using the UNIQUE Constraint
-- 1
CREATE TABLE SCHOOL (
  s_name VARCHAR(50) UNIQUE
);
-- 2
CREATE TABLE COLLAGE (
s_name VARCHAR(50),
s_address VARCHAR(100),
  CONSTRAINT s_name_un UNIQUE(s_name)
);

-- 3
ALTER TABLE COLLAGE ADD COLUMN s_dep VARCHAR(50);

ALTER TABLE COLLAGE ADD CONSTRAINT s_dep_un UNIQUE(s_dep);

-- Using the CHECK Constraint
CREATE TABLE BALL (
  b_color CHAR(1),
  CONSTRAINT b_check_color CHECK (b_color IN ('B', 'W'))
);

-- NOT NULL Constraint (3)Examples:
-- 1
CREATE TABLE NN1 (
  b_color CHAR(1) NOT NULL
);

-- 2
ALTER TABLE NN1
ADD COLUMN new_col VARCHAR(30);
ALTER TABLE NN1
MODIFY COLUMN new_col VARCHAR(30) NOT NULL;
-- Display constraint listing for a specific table
SELECT 
    CONSTRAINT_NAME, 
    CONSTRAINT_TYPE, 
    TABLE_NAME, 
    TABLE_SCHEMA
FROM 
    INFORMATION_SCHEMA.TABLE_CONSTRAINTS
WHERE 
    TABLE_SCHEMA = 'UniversityDB' 
    AND TABLE_NAME = 'courses';
    
    -- Disabling / Enabling Constraints
    











