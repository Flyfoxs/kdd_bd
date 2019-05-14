import mysql.connector

from core.feature import *

import socket
mysql_pass = 'Had00p!!'
redis_pass = 'cisco2016redis'



from redlock import RedLockFactory
factory = RedLockFactory(
    connection_details=[

        {'host': '10.224.38.31', 'port': 8690, 'db': 13, 'password':redis_pass},
        #{'host': '10.224.38.43','port': 8690, 'db': 13, 'password':redis_pass },

    ])





def get_connect():
    db = mysql.connector.connect(user='ai_lab', password=mysql_pass,
                                 host='vm-ai-2',
                                 database='ai')
    return db

# def get_session():
#     session = mysqlx.get_session({
#         'host': 'vm-ai-2',
#         #'port': 3306,
#         'user': 'ai_lab',
#         'password': mysql_pass,
#         'schema': 'ai'
#     })
#
#     return session

def log_begin(version, drop_columns):
    try:

        db = get_connect()


        server = socket.gethostname()


        sql = """insert into search_para(
                    version  ,
                    drop_columns,
                     server)
                      values
                    (
     
                    '{version}'	 ,
                    '{drop_columns}',
                    '{server}'	  
                       )
                        """.format(drop_columns=drop_columns, version=version, server=server)
        cur = db.cursor()
        logger.info(sql)
        cur.execute(sql)
        db.commit()
        return True
    except Exception as e:
        logger.exception(e)
        logger.warning(f'Do not get the lock:{e}')
        return False


def log_end(version, drop_columns, feature_nums,  score):
    db = get_connect()
    server = socket.gethostname()

    sql = """update search_para 
               set feature_nums = {feature_nums} ,
                   score = {score} ,
                   end_time = now(),
                   cost = now() - start_time
               where 
                       server =  '{server}' and
                       version =  '{version}' and
                       drop_columns =  '{drop_columns}' and
                       score is null
                       """.format(drop_columns=drop_columns,
                                  version=version,
                                  server=server,
                                  feature_nums=feature_nums,
                                  score=score)
    cur = db.cursor()
    logger.debug(sql)
    cur.execute(sql)
    db.commit()


if __name__ == '__main__':
    log_begin('bdx', 'xx')

    log_end('bdx', 'xx', 10, 10, 0.2)