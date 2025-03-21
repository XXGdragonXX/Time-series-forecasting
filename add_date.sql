-- add purchase date for to the existing data for 3 months
UPDATE purchase_data
SET purchase_date = DATE_ADD('2025-01-01', INTERVAL FLOOR(1 + (RAND() * 90)) DAY);