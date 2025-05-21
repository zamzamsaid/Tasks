CREATE DATABASE Textbooks;
USE Textbooks;

CREATE TABLE student (
St_ID INT(6) PRIMARY KEY,
St_Major VARCHAR(30),
St_Cohort INT (4) NOT NULL,
CONSTRAINT s_st_id CHECK (St_ID >0)
);

CREATE TABLE employee (
Em_ID INT(6) PRIMARY KEY,
Em_Office VARCHAR(4) NOT NULL,
Em_Ext INT (4),
CONSTRAINT e_em_id CHECK (Em_ID >0),
CONSTRAINT e_em_EXT CHECK (Em_Ext >1000)
);

CREATE TABLE college(
Cl_Code VARCHAR(3) PRIMARY KEY,
Cl_Name VARCHAR(40) NOT NULL,
Cl_Dean  VARCHAR (30)
);

CREATE TABLE department (
    Dp_Code VARCHAR(4) PRIMARY KEY,
    Dp_Name VARCHAR(40) NOT NULL,
    Dp_HoD VARCHAR(30),
    Dp_Col VARCHAR(3),
    FOREIGN KEY (Dp_Col) REFERENCES college(Cl_Code)
);

CREATE TABLE borrower(
    Br_ID NUMERIC(6) PRIMARY KEY,
    Br_Name VARCHAR(30) NOT NULL,
    Br_Dept VARCHAR(4),
    Br_Mobile  VARCHAR(8),
    Br_City VARCHAR(20),
    Br_House VARCHAR(4),
    Br_Type VARCHAR(1),
    CONSTRAINT b_br_id CHECK (Br_ID > 0),
    FOREIGN KEY (Br_Dept) REFERENCES department(Dp_Code),
    CONSTRAINT b_br_mobile CHECK (Br_Mobile >= '90000000'),
    CONSTRAINT b_br_type CHECK (Br_Type IN ('S', 'E'))
);

CREATE TABLE BOOK (
    Bk_ID NUMERIC(6) PRIMARY KEY,
    Bk_Title VARCHAR(50) NOT NULL,
    Bk_Edition NUMERIC(2),
    Bk_ofPages INT(4),
    Bk_TotalCopies NUMERIC(5),
    Bk_RemCopies NUMERIC(5),
    CONSTRAINT b_bk_id CHECK (Bk_ID > 0),
    CONSTRAINT b_bk_pages CHECK (Bk_ofPages > 0)
);

CREATE TABLE BookTopic (
    Tp_BkID NUMERIC(6),
    Tp_Desc VARCHAR(30) NOT NULL,
    FOREIGN KEY (Tp_BkID) REFERENCES BOOK(Bk_ID)
);

CREATE TABLE course (
    Cr_Code VARCHAR(8) PRIMARY KEY,
    Cr_Title VARCHAR(40) NOT NULL,
    Cr_CH NUMERIC(2),
    Cr_ofSec INT(2),
    Cr_Dept VARCHAR(4),
    CONSTRAINT c_cr_ch CHECK (Cr_CH > 0),
    CONSTRAINT c_cr_ofsec CHECK (Cr_ofSec > 0),
  
    FOREIGN KEY (Cr_Dept) REFERENCES department(Dp_Code)
);

CREATE TABLE link (
    Li_CrCode VARCHAR(8),
    Li_BkID NUMERIC(6),
    Li_usage VARCHAR(1),
    CONSTRAINT l_usage CHECK (Li_usage IN ('T', 'R')),
    FOREIGN KEY (Li_CrCode) REFERENCES course(Cr_Code),
    FOREIGN KEY (Li_BkID) REFERENCES book(Bk_ID)
);

CREATE TABLE Registration (
    Re_BrID NUMERIC(6) ,
    Re_CrCode CHAR(8),
    Re_Semester CHAR(6) NOT NULL,
    CONSTRAINT fk_Re_BrID FOREIGN KEY (Re_BrID) REFERENCES borrower(Br_ID),
    CONSTRAINT fk_Re_CrCode FOREIGN KEY (Re_CrCode) REFERENCES course(Cr_Code)
);

CREATE TABLE Issuing (
    is_BrID NUMERIC(6),
    is_BkID NUMERIC(6),
    is_DateTaken DATE NOT NULL,
    is_DateReturn DATE,
    CONSTRAINT fk_is_BrID FOREIGN KEY (is_BrID) REFERENCES borrower(Br_ID),
    CONSTRAINT fk_is_BkID FOREIGN KEY (is_BkID) REFERENCES book(Bk_ID),
    CONSTRAINT chk_DateReturn_gt_Taken CHECK (is_DateReturn > is_DateTaken)
);

INSERT INTO course VALUES('BIOL1000', 'Intro. To Biology', 3, 5,'BIOL');

INSERT INTO course VALUES('CHEM2000', 'Advanced Chemistry', 2, 2,'CHEM');


INSERT INTO college VALUES('COM', 'Economy', 'Prof. Fahim');

INSERT INTO college VALUES('SCI', 'Science', 'Prof. Salma');

INSERT INTO college VALUES('EDU', 'Education', 'Dr. Hamad');

INSERT INTO college VALUES('ART', 'Arts', 'Dr. Abdullah');

INSERT INTO college (Cl_Code, Cl_Name) VALUES
('COM', 'College of Commerce'),
('SCI', 'College of Science'),
('EDU', 'College of Education');
INSERT INTO department VALUES('INFS','Information Systems','Dr. Kamla','COM');

SELECT Cl_Code,Cl_Name FROM college;

INSERT INTO borrower VALUES (92120,'Ali','INFS',99221133,'Seeb','231','S');

INSERT INTO borrower VALUES (10021,'Said','INFS',91212129,'Seeb','100','S');

INSERT INTO borrower VALUES (10023,'Muna','FINA', NULL, 'Barka','12','S');

INSERT INTO borrower VALUES (3000,'Mohammed','COMP',90000009,'Seeb','777','E');

INSERT INTO borrower VALUES (4000,'Nasser','INFS',99100199,'Sur','11','E');

INSERT INTO student VALUES(92120,'INFS',2012);

INSERT INTO student VALUES(10021,'INFS',2015);

INSERT INTO student VALUES(10023,'FINA',2015);

INSERT INTO employee VALUES(3000,'12',2221);

INSERT INTO employee VALUES(4000,'15',1401);

INSERT INTO department VALUES ('COMP', 'Computer Science', 'Dr. Kamla', 'COM');
INSERT INTO department VALUES ('BIOL', 'Biology', 'Dr. Ahmad', 'SCI');
INSERT INTO department VALUES ('CHEM', 'Chemistry', 'Dr. Nora', 'SCI');
INSERT INTO course VALUES ('COMP4201', 'Database1', 3, 1, 'COMP');
INSERT INTO course VALUES ('COMP4202', 'Database2', 3, 2, 'COMP');
INSERT INTO course VALUES ('BIOL1000', 'Intro. To Biology', 3, 5, 'BIOL');
INSERT INTO course VALUES ('CHEM2000', 'Advanced Chemistry', 2, 2, 'CHEM');




INSERT INTO book VALUES(1001,'Database1',2,450,150,65);

INSERT INTO book VALUES(1002,'Database2',3,300,100,100);

INSERT INTO book VALUES(2001,'Intro. to Finanace',1,300,75,40);

INSERT INTO book VALUES(3001,'Basic Op Mgmt',1,320,100,77);

INSERT INTO BOOK (Bk_ID, Bk_Title, Bk_Edition, Bk_ofPages, Bk_TotalCopies, Bk_RemCopies) VALUES
(1001, 'Database Basics', 1, 200, 10, 5),
(1002, 'Advanced SQL', 2, 300, 15, 10),
(1003, 'Business Management', 1, 250, 8, 4);

INSERT INTO bookTopic VALUES (1001,'Basic DB Skills');

INSERT INTO bookTopic VALUES (1001,'ERD');

INSERT INTO bookTopic VALUES (1001,'EERD');

INSERT INTO bookTopic VALUES (1002,'SQL');

INSERT INTO bookTopic VALUES (1002,'Pl/SQL');

INSERT INTO bookTopic VALUES (3001,'Management Skills');

INSERT INTO employee VALUES (3001,9987 ,2356);
-- Confirm the record exists
SELECT * FROM employee WHERE Em_ID = 3000;

-- Run the update
UPDATE employee
SET Em_Office = 20,
    Em_Ext = 3331
WHERE Em_ID = 3001;
-- Verify the update
SELECT * FROM employee WHERE Em_ID = 3000;

SET @St_ID = 12345;

SET @St_Cohort = 2024;
 
INSERT INTO student (St_ID, St_Cohort)

VALUES (@St_ID, @St_Cohort);
 
SELECT * FROM student;

SELECT * FROM book; 
UPDATE book SET Bk_ofPages = Bk_ofPages-50 WHERE BK_ID = 1001;
SELECT * FROM book; 

-- Insert department first if missing
INSERT INTO department (Dp_Code, Dp_Name, Dp_HoD, Cl_Code)
VALUES ('COMP', 'Computer Science', 'Dr. Smith', 'COM');

INSERT INTO course VALUES('COMP4201', 'Database1', 3, 1,'COM');
INSERT INTO department VALUES('HIST','History','Dr. Said','EDU');
INSERT INTO department VALUES('CHEM', 'Chemistry', 'Dr. Alaa', 'SCI');

-- UPDATE using subquery IT WILL CHANGE THE VALUE OF SECONED ROW WITH THE FIRST ROW MENTION IN THE CODE
SELECT * FROM book; 

UPDATE book b2
JOIN book b1 ON b1.Bk_ID = 1003
SET b2.Bk_TotalCopies = b1.Bk_TotalCopies
WHERE b2.Bk_ID = 3002;

-- DELETE
DELETE FROM book 
WHERE Bk_ID = 1002;

SELECT * FROM book; 

-- select with groupby
-- Which colleges have more than one department, and how many departments does each of those colleges have?"
SELECT Dp_Col, COUNT(*) AS Dept_Count
FROM department
GROUP BY Dp_Col
HAVING COUNT(*) > 1
ORDER BY Dept_Count DESC;

-- "How many colleges are there for each dean’s name, and only show deans who are responsible for more than one college?"
SELECT Cl_Dean, COUNT(*) AS Number_of_Colleges -- The * in COUNT(*) means "count all rows" in each group
FROM college
GROUP BY Cl_Dean
HAVING COUNT(*) > 1
ORDER BY Number_of_Colleges ASC;

SELECT Bk_Title , Bk_RemCopies *3+1 FROM book;

-- Displaying Distinct Rows (it avoid the repeated values so befor use i have dr.kamla two times when use it it display once only
SELECT DISTINCT Dp_HoD 
FROM department;

-- select specific number of rows
SELECT * FROM department LIMIT 3; -- GIVES ME 3 ROWS
SELECT Dp_Name FROM department LIMIT 3; -- HERE TAKE 3 ROWS FROM Dp_Name COLUMN
SELECT Dp_Name,Dp_HoD FROM department LIMIT 3; -- HERE TAKE 3 ROWS FROM Dp_Name,Dp_HoD COLUMN

-- Comparing Values
-- • Retrieve all employee whose Em_Office is not equal to 20
SELECT * FROM employee
WHERE Em_Office <> 20;

CREATE TABLE customer (
  CUSTOMERID INT PRIMARY KEY,         
  FIRSTNAME VARCHAR(50) NOT NULL,     
  LASTNAME VARCHAR(50) NOT NULL,     
  DOB DATE NOT NULL,                 
  PHONE VARCHAR(15)              
);

INSERT INTO customer (CUSTOMERID, FIRSTNAME, LASTNAME, DOB, PHONE)
VALUES (1, 'Ali', 'Khan', '1995-06-15', '0501234567');

INSERT INTO customer (CUSTOMERID, FIRSTNAME, LASTNAME, DOB, PHONE)
VALUES (2, 'Sara', 'Ahmed', '1998-09-21', '0507654321');

INSERT INTO customer (CUSTOMERID, FIRSTNAME, LASTNAME, DOB, PHONE)
VALUES (3, 'John', 'Smith', '1990-01-05', '0559876543');

INSERT INTO customer (CUSTOMERID, FIRSTNAME, LASTNAME, DOB, PHONE)
VALUES (4, 'Fatima', 'Hassan', '1992-03-12', '0561112233');

INSERT INTO customer (CUSTOMERID, FIRSTNAME, LASTNAME, DOB, PHONE)
VALUES (5, 'Mohammed', 'Ali', '1988-12-30', '0577778888');

SELECT * FROM customer;

SELECT * FROM customer
WHERE CUSTOMERID > ANY (SELECT CUSTOMERID FROM 
customer WHERE CUSTOMERID BETWEEN 2 AND 4);

SELECT * FROM customer
WHERE CUSTOMERID > ALL (SELECT CUSTOMERID FROM 
customer WHERE CUSTOMERID BETWEEN 2 AND 4);

-- using SQL operators
-- (_o%) -> _ this mean one character it take only one before "o" , exactly one character
-- % mean many character so after "o" can be zero or many character 
SELECT * FROM customer
WHERE FIRSTNAME LIKE '_o%';

-- 
CREATE TABLE product_types (
    product_type_id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(8, 2),
    quantity_in_stock INT,
    product_type_id INT,
    expiry_date DATE,
    added_by VARCHAR(50),
    FOREIGN KEY (product_type_id) REFERENCES product_types(product_type_id)
);
INSERT INTO product_types (product_type_id, name, description)
VALUES 
(101, 'Beverages', 'All kinds of drinks'),
(102, 'Snacks', 'Packaged snack foods'),
(103, 'Dairy', 'Milk and dairy products');

INSERT INTO products (product_id, name, price, quantity_in_stock, product_type_id, expiry_date, added_by)
VALUES 
(1, 'Apple Juice', 3.99, 100, 101, '2025-06-01', 'Admin1'),
(2, 'Cheddar Cheese', 6.50, 50, 103, '2024-12-01', 'Admin2'),
(3, 'Chocolate Bar', 1.99, 200, 102, '2025-03-15', 'Admin1');

-- Performing SELECT Statements That Use Two or more Tables
-- inner join
SELECT p.name, pt.name 
FROM products p , product_types pt
WHERE p.product_type_id = pt.product_type_id
ORDER BY p.name;
-- outer join
SELECT products.product_id, products.name, product_types.name
FROM products
LEFT JOIN product_types ON products.product_type_id = product_types.product_type_id;

-- Using Concatenation
CREATE TABLE details (
firstName VARCHAR(30),
lastName VARCHAR(30),
address VARCHAR(30)
);

INSERT INTO details (firstName, lastName, address) VALUES
('John', 'Doe', 'USA'),
('Jane', 'Smith', 'Canada'),
('Alice', 'Johnson', 'Australia');

SELECT CONCAT(firstName, ' ', lastName, ' live in ', address) AS full_info
FROM details;

SELECT Cl_Dean, COUNT(Cl_Name) AS CLNAME
FROM college
GROUP BY Cl_Dean
HAVING COUNT(Cl_Name) > 1
ORDER BY CLNAME ASC;
-- count how many user borrow books
SELECT COUNT(Br_Name)
FROM borrower;

SELECT COUNT(Br_ID) 
FROM borrower
WHERE Br_Type;

SELECT * FROM course;
SELECT * FROM department;

SELECT * FROM course 
JOIN department ON (Dp_Code = Cr_Dept);
-- INNER JOIN WITH THREE TABLES
SELECT * FROM department 
JOIN course ON (Dp_Code = Cr_Dept)
JOIN borrower ON (Br_Dept = Dp_Code);

-- RIGHT JOIN
SELECT Br_Name,Dp_Name
FROM borrower
RIGHT JOIN department
ON (Br_Dept = Dp_Code);

-- LEFT JOIN
SELECT Cr_Dept,Dp_Name,
COUNT (Cr_Code)
FROM course
RIGHT JOIN department
ON (Cr_Dept = Dp_Code);

-- 
SELECT  Dp_Code ,Dp_Name,
COUNT(Cr_Code) 
FROM course
RIGHT JOIN department 
ON (Cr_Dept = Dp_Code)
GROUP BY Dp_Code,Dp_Name;
-- HERE RETURN ONLY THE COMMON
SELECT Dp_Name, COUNT(Cr_Code)
FROM course
JOIN department 
ON Cr_Dept = Dp_Code
GROUP BY Dp_Name;

-- IF I WANT COUNT CORUSE IN EACH DEPARTMENT BUT SHOULD SHOW ME EVEN WHEN DEPARTMENT NULL 
-- I WAILL USE LEFT JOIN
SELECT Dp_Name, COUNT(Cr_Code)
FROM department
LEFT JOIN course 
ON Cr_Dept = Dp_Code
GROUP BY Dp_Name;

