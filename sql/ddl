
drop table search_para;

create table search_para(
ID int NOT NULL AUTO_INCREMENT PRIMARY KEY ,
version VARCHAR(16),
drop_columns VARCHAR(512),
feature_nums int,
start_time datetime DEFAULT NOW() ,
end_time datetime  ,
cost int,
score DECIMAL(8,6),
server VARCHAR(16)
);


CREATE UNIQUE INDEX sp_pk on search_para(version, drop_columns);