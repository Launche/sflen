--建表语句

drop table if exists aimp_deep_ctr;

CREATE TABLE aimp_deep_ctr (
  algo varchar(255) NOT NULL ,
  data_type varchar(255) DEFAULT NULL ,
  epochs varchar(255) DEFAULT NULL ,
  optimizer varchar(255) DEFAULT NULL ,
  dropout varchar(255) DEFAULT NULL ,
  log_loss  varchar(255) DEFAULT NULL ,
  auc  varchar(255) DEFAULT NULL ,
  insert_time timestamp DEFAULT NULL
) ;
