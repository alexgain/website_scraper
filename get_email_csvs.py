import numpy as np
import email, getpass, imaplib, os, re
import matplotlib.pyplot as plt
import re

def return_last_csv(id_ = -1):
    def get_urls(string):
        return re.findall('(https://\S+)', string)    
    
    detach_dir = "./"
    # user = input("Enter your GMail username --> ")
    # pwd = getpass.getpass("Enter your password --> ")
    user = 'alexzgain'
    pwd = 'Shoogiebaba23'
    
    m = imaplib.IMAP4_SSL("imap.gmail.com")
    m.login(user, pwd)
    
    m.select("inbox") 
    
    type, data = m.search(None, '(FROM "stryd@stryd.com")')
    # type, data = m.search(None, 'ALL')
    mail_ids = data[0]
    id_list = mail_ids.split()
    
    urls = []
    for i in id_list:
        typ, data = m.fetch(i, '(RFC822)' )
        msg = email.message_from_bytes(data[0][1])
    
        ##getting urls from body:
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True)
                urls = urls + get_urls(str(body))
            else:
                continue
        
    return np.loadtxt(urls[id_][:len(urls[id_])-1],delimiter=',')

            
def get_len():
    def get_urls(string):
        return re.findall('(https://\S+)', string)    
    
    detach_dir = "./"
    # user = input("Enter your GMail username --> ")
    # pwd = getpass.getpass("Enter your password --> ")
    user = 'alexzgain'
    pwd = 'Shoogiebaba23'
    
    m = imaplib.IMAP4_SSL("imap.gmail.com")
    m.login(user, pwd)
    
    m.select("inbox") 
    
    type, data = m.search(None, '(FROM "stryd@stryd.com")')
    # type, data = m.search(None, 'ALL')
    mail_ids = data[0]
    id_list = mail_ids.split()
    
    urls = []
    for i in id_list:
        typ, data = m.fetch(i, '(RFC822)' )
        msg = email.message_from_bytes(data[0][1])
    
        ##getting urls from body:
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True)
                urls = urls + get_urls(str(body))
            else:
                continue
        
    return len(urls)

