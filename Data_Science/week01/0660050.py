#!/usr/bin/python3

# Import lib
import numpy as np
import requests
import sys
import math
import time
from bs4 import BeautifulSoup
import functools

# Problem 1
# get real article year need too much time

'''def get_article_year(url):
    r = requests.get(url)
    content = r.text
    page_soup = BeautifulSoup(content, "html.parser")
    articles_class = page_soup.find_all(class_="article-metaline")
    
    article_time = ""
    find_time = False
    find_sendmail = False
    for article in articles_class:
        if article.find("span", "article-meta-tag").string.find("時間"):
            find_time = True
        elif find_time:
            article_time = article.find("span", "article-meta-value").string
    article_time = str(article_time)
    year_split = article_time.split(" ")
    
    # check send email station
    all_paragraph = page_soup.find_all(("div", {"id": "main-content"}))
    for paragraph in all_paragraph: 
        #print (type(paragraph))
        paragraph_split = paragraph.text.split('\n')
        for line in paragraph_split:
            #print (line)
            if line.find("發信站") is not -1:
                find_sendmail = True
                break
    if not find_sendmail:
        year_split = "-1"
    return year_split[ len(year_split)-1 ]
'''
def valid_url(url):
    if url.find("https://www.ptt.cc/bbs/Beauty/M.1490936972.A.60D.html") is not -1:
        return False
    elif url.find("https://www.ptt.cc/bbs/Beauty/M.1494776135.A.50A.html") is not -1:
        return False
    elif url.find("https://www.ptt.cc/bbs/Beauty/M.1503194519.A.F4C.html") is not -1:
        return False
    elif url.find("https://www.ptt.cc/bbs/Beauty/M.1504936945.A.313.html") is not -1:
        return False
    elif url.find("https://www.ptt.cc/bbs/Beauty/M.1505973115.A.732.html") is not -1:
        return False
    elif url.find("https://www.ptt.cc/bbs/Beauty/M.1507620395.A.27E.html") is not -1:
        return False
    elif url.find("https://www.ptt.cc/bbs/Beauty/M.1510829546.A.D83.html") is not -1:
        return False
    elif url.find("https://www.ptt.cc/bbs/Beauty/M.1512141143.A.D31.html") is not -1:
        return False
    else:
        return True

def crawl(url, index):
    global file_articles, file_popular, find_2017, find_2018, article_counter
    print ("now crawling url = ", url)
    ptt_url = "https://www.ptt.cc"
    r = requests.get(url)
    content = r.text
    page_soup = BeautifulSoup(content, "html.parser")
    
    # Find articles
    articles_class = page_soup.find_all(class_="r-ent")
    for num in range(0, len(articles_class)):
        article = articles_class[num]
        if article.a is not None:
            if article.a.text[0:4].find("公告") == -1:
                article_counter += 1
                
                # Find push counter
                push_counter_string = article.find("div", "nrec").string
                if push_counter_string is None:
                    push_counter_string = "0"
                
                # Find title and url
                title = article.a.text
                article_url = article.a.get('href')
                article_url = ptt_url + article_url
                
                # the date of article
                date = article.find("div", "date").string

                #date = date.split("/")[0]+date.split("/")[1]
                
                # Check year

                if not find_2017 and date.find("1/01") is not -1:
                    find_2017 = True
                elif find_2017  and date.find("1/01") is not -1 and article_counter >= 1000:
                    find_2018 = True

                #year = get_article_year(article_url) 

                if find_2018:
                    return

                # Get article info (date, title, url)
                if find_2017  and valid_url(article_url):
                    date = date.split("/")[0]+date.split("/")[1]
                    date = date.replace(" ", "")
                    article_info = date + "," + title + "," + article_url
                    
                    file_articles.write(article_info+'\n')
                    if push_counter_string.find("爆") is not -1 :
                        file_popular.write(article_info+'\n')
                    #print (date + " " + title + " " + article_url)

    
    # Find previous page url 
    time.sleep(0.2)

    # using index to accelerate
    '''
    page_button_class = page_soup.find_all(class_="btn wide")
    for page_button in page_button_class:
        if page_button.text is not None:
            if page_button.text.find("上頁") != -1:
                previos_page_url = page_button.get('href')
                previos_page_url = ptt_url + previos_page_url
                print (previos_page_url)
                crawl(previos_page_url, ndex)
    '''
    
    index += 1
    previos_page_url = ptt_url + "/bbs/Beauty/index" + str(index) + ".html"
    crawl(previos_page_url, index)


# Problem 2
def find_push_and_boo(start_date, end_date, dic, like_top10, boo_top10):
    #file_articles = open('all_articles.txt', 'r', encoding = 'utf8') 
    file_articles = open('all_articles.txt', 'r', encoding = 'utf8')
    lines = file_articles.readlines()
    for line in lines:
        line_split = line.split(",")
        if int(line_split[0]) >=start_date and int(line_split[0])<=end_date:
            url = line_split[len(line_split)-1][:-1]
            dic = search_push_and_boo(url, dic)
            time.sleep(0.1)

    like_top10, boo_top10 = find_top_10(dic, like_top10, boo_top10)
    return dic, like_top10, boo_top10
        
def search_push_and_boo(url, dic):
    global push_counter, boo_counter
    print (url)
    r = requests.get(url)
    content = r.text
    page_soup = BeautifulSoup(content, "html.parser")
    push_content = page_soup.find_all(class_ = "push")
    for paragraph in push_content:
        push_tag = paragraph.find("span", "push-tag")
        push_user = paragraph.find("span", "f3 hl push-userid")
        if push_tag is not None and push_user is not None:
            if dic.get(push_user.text) is None:
                dic[push_user.text] = [-1, -1]
            if push_tag.text.find("推") is not -1:
                push_counter += 1
                dic[push_user.text][0] += 1
            elif push_tag.text.find("噓") is not -1:
                boo_counter += 1
                dic[push_user.text][1] += 1
    return dic

def find_top_10(dic, like_top10, boo_top10):
    for person in dic:
        
        if len(like_top10) < 11:
            like_top10.append( [person, dic[person][0]] )
        else:
            like_top10 = sorted(like_top10, key=functools.cmp_to_key(is_bigger))
            target = [person, dic[person][0]]
            if is_bigger(like_top10[len(like_top10)-1], target):
                like_top10[len(like_top10)-1] = target
    
        if len(boo_top10) < 11:
            boo_top10.append( [person, dic[person][1]] )
        else:
            boo_top10 = sorted(boo_top10, key=functools.cmp_to_key(is_bigger))
            #print (boo_top10)
            target = [person, dic[person][1]]
            if is_bigger(boo_top10[len(boo_top10)-1], target):
                boo_top10[len(boo_top10)-1] = target    
    return like_top10, boo_top10
    
def is_bigger(source, target):
    source_person = source[0]
    source_value = source[1]
    target_person = target[0]
    target_value = target[1]
    if source_value < target_value:
        return 1
    elif source_value > target_value:
        return -1
    else:
        index = 0
        while index < len(source_person) and index < len(target_person):
            # ex: a > A, A > 1
            if ord(source_person[index]) > ord(target_person[index]):
                return 1
            elif ord(source_person[index]) < ord(target_person[index]):
                return -1
            else:
                index += 1
        if len(source_person) > len(target_person):
            return 1
        else:
            return -1

# Problem 3
def find_popular(start_date, end_date, file_popular_picture):
    file_articles = open('all_popular.txt', 'r', encoding = 'utf8')
    lines = file_articles.readlines()
    popular_counter = 0
    pics_url = []
    for line in lines:
        line_split = line.split(",")
        if int(line_split[0]) >=start_date and int(line_split[0])<=end_date:
            popular_counter += 1
            url = line_split[2][:-1]
            find_jpg_in_page (url, pics_url)
            time.sleep(0.2)
        
    file_popular_picture.write("number of popular articles: " + str(popular_counter) + "\n")
    for pic_url in pics_url:
        file_popular_picture.write(pic_url + "\n")    

def find_jpg_in_page(url, pics_url):
    r = requests.get(url)
    print (url)
    content = r.text
    page_soup = BeautifulSoup(content, "html.parser")
    main_content = page_soup.find_all('a', href=True)
    for paragraph in main_content:
        if paragraph is not None:
            href_content = paragraph.text
            #if ".jpg" in href_content or ".jpeg" in href_content or ".png" in href_content or ".gif" in href_content:
            if ".jpg" in href_content[-4:] or ".jpeg" in href_content[-5:] or ".png" in href_content[-4:] or ".gif" in href_content[-4:]:
                # display picture by url
                # display(Image(href_content))
                pics_url.append(href_content)

# Problem 4
def find_keyword(start_date, end_date, file_keyword_picture, keyword):
    file_articles = open('all_articles.txt', 'r', encoding = 'utf8')
    lines = file_articles.readlines()
    for line in lines:
        line_split = line.split(",")
        if int(line_split[0]) >=start_date and int(line_split[0])<=end_date:
            url = line_split[2][:-1]
            pics_url = []
            if is_keyword_in_page (url, keyword):
                print ("Contain keyword url = ", url)
                find_jpg_in_page(url, pics_url)
                time.sleep(0.2)
                for pic_url in pics_url:
                    file_keyword_picture.write(pic_url + "\n") 

        
def is_keyword_in_page(url, keyword):
    r = requests.get(url)
    print (url)
    content = r.text
    page_soup = BeautifulSoup(content, "html.parser")    
    main_content = page_soup.find_all("div", {"id": "main-content"})
    find_keyword = False
    for paragraph in main_content: 
        #print (type(paragraph))
        paragraph_split = paragraph.text.split('\n')
        for line in paragraph_split:
           # print (line)
            if line.find("發信站") is not -1:
                break
            elif line.find(keyword) is not -1:
                find_keyword =  True
    #print (*pics_url, sep='\n')
    return find_keyword


if __name__ == "__main__":

    function = sys.argv[1]

    if function == "crawl":
        file_articles = open('all_articles.txt', 'w', encoding = 'utf8')
        file_popular = open('all_popular.txt', 'w', encoding = 'utf8')
        find_2017 = False
        find_2018 = False
        article_counter = 0
        url = "https://www.ptt.cc/bbs/Beauty/index1991.html"  
        #url = "https://www.ptt.cc/bbs/Beauty/index2127.html"
        crawl(url, 1991)
        file_articles.close()
        file_popular.close()
        print (function, " Done")

    elif function == "push":
        start_date = int(sys.argv[2])
        end_date = int(sys.argv[3])
        dic = {} 

        push_counter = 0
        boo_counter = 0
        like_top10 = []
        boo_top10 = []
        dic, like_top10, boo_top10 = find_push_and_boo(start_date, end_date, dic, like_top10, boo_top10)

        f_name = "push[" + str(start_date) + "-" + str(end_date) + "].txt"
        file_push_and_boo = open(f_name, 'w')
        file_push_and_boo.write("all like: " + str(push_counter) + "\n") 
        file_push_and_boo.write("all boo: " + str(boo_counter) + "\n") 
        print ("all like: " + str(push_counter) + "\n")
        print ("all boo: " + str(boo_counter) + "\n") 
        for index in range(0, len(like_top10)-1):
            text = "like #" + str(index+1) + ": " + like_top10[index][0] + " " + str(like_top10[index][1]+1)
            file_push_and_boo.write(text + "\n") 
            print (text)
        for index in range(0, len(boo_top10)-1):
            text = "boo #" + str(index+1) + ": " + boo_top10[index][0] + " " + str(boo_top10[index][1]+1)
            file_push_and_boo.write(text + "\n") 
            print (text)

        file_push_and_boo.close()
        print (function, "Done")

    elif function == "popular":
        start_date = int(sys.argv[2])
        end_date = int(sys.argv[3])

        f_name = "popular[" + str(start_date) + "-" + str(end_date) + "].txt"
        file_popular_picture = open(f_name, 'w')
        find_popular(start_date, end_date, file_popular_picture)
        file_popular_picture.close()

        print (function, " Done")


    elif function == "keyword":
        keyword = sys.argv[2]
        start_date = int(sys.argv[3])
        end_date = int(sys.argv[4])        
        f_name = "keyword(" + keyword + ")[" + str(start_date) + "-" + str(end_date) + "].txt"
        file_keyword_picture = open(f_name, 'w')
        find_keyword(start_date, end_date, file_keyword_picture, keyword)
        file_keyword_picture.close()
        print (function, " Done")