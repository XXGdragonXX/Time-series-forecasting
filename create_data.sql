-- creating the final input with date at product level to predict the next month forecast for each product based on last 3 months sale 

-- Create the FIRST view: purchase level data

CREATE VIEW purchase_lvl_data AS 
select 
pd.Product_Category as `Brand`,
pd.purchase_date as `Date`,
sum(pd.Purchase) as `Total_Sale`
from purchase_data pd 
GROUP BY 
pd.Product_Category,
pd.purchase_date;

-- Create the second view: monthly_sale_jan

CREATE VIEW monthly_sale_jan AS
select 
Brand,
sum(Total_Sale) as `Jan_Sale`
from purchase_lvl_data
WHERE 
MONTH(Date) = 1
GROUP BY Brand;

-- create veiw for sale for february
CREATE VIEW monthly_sale_feb AS
select 
Brand,
sum(Total_Sale) as `feb_Sale`
from purchase_lvl_data
WHERE 
MONTH(Date) = 2
GROUP BY Brand;

-- create veiw for sale for March
CREATE VIEW monthly_sale_march AS
select 
Brand,
sum(Total_Sale) as `March_Sale`
from purchase_lvl_data
WHERE 
MONTH(Date) = 2
GROUP BY Brand;


-- Create the final table
CREATE TABLE FINAL_TABLE 
SELECT 
jan.Brand as `Brand`,
jan.Jan_Sale as `Jan_Sale`,
feb.feb_Sale as `Feb_SALE`,
march.March_Sale as `March_Sale`

from 
monthly_sale_jan jan
JOIN
monthly_sale_feb feb
ON
jan.Brand = feb.Brand
JOIN
monthly_sale_march march 
ON 
jan.Brand = march.Brand