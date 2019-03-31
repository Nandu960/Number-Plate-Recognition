from configparser import ConfigParser
from mysql.connector import MySQLConnection, Error

 
def read_db_config(filename='config.ini', section='mysql'):
    """ Read database configuration file and return a dictionary object
    :param filename: name of the configuration file
    :param section: section of database configuration
    :return: a dictionary of database parameters
    """
    # create parser and read ini configuration file
    parser = ConfigParser()
    parser.read(filename)
 
    # get section, default to mysql
    db = {}
    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            db[item[0]] = item[1]
    else:
        raise Exception('{0} not found in the {1} file'.format(section, filename))
 
    return db


def connect():
    """ Connect to MySQL database """
 
    db_config = read_db_config()
 
    try:
        print('Connecting to MySQL database...')
        conn = MySQLConnection(**db_config)
 
        if conn.is_connected():
            print('connection established.')
            cursor=conn.cursor();
            cursor.execute("create table reg_logs(car_number_plate varchar(20),time varchar(20),date varchar(20),description varchar(20))")
        else:
            print('connection failed.')
 
    except Error as error:
        print(error)
 
    finally:
        conn.close()
        print('Connection closed.')

def insert_numplate(car_number_plate,time,date,description,q):
    q=q+"(\""+car_number_plate+"\",\""+time+"\",\""+date+"\",\""+description+"\")"
    
 
    try:
        db_config = read_db_config()
        conn = MySQLConnection(**db_config)
 
        cursor = conn.cursor()
        cursor.execute(q)
 
        conn.commit()
    except Error as error:
        print(error)
 
    finally:
        cursor.close()
        conn.close();

query="insert into reg_logs(car_number_plate,time,date,description) values "
c=0
##inpt=open("numinput.txt","r")
##inp=inpt.readlines()

def add(x,y,z):

    r=""
    if x[:1]=="AP":
        r="https://aprtacitizen.epragathi.org/#!/vehicleRegistrationSearch"
    elif x[:1]=="UP":
        r="https://vahan.nic.in/nrservices/faces/user/jsp/SearchStatus.jsp"
    
    insert_numplate(x,y,z,r,query)

##
##for i in inp:
##    i=i.replace("\n","")
##    j=i.split(" ")
##
##    if (c==0):
##        connect()
##        insert_numplate(j[0],j[1],j[2],query)
##        c+=1
##    else:
##        insert_numplate(j[0],j[1],j[2],query)
