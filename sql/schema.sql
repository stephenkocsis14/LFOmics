-- Database: multiomics

CREATE TABLE results_files (
    file_id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_content LONGBLOB,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
