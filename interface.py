# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:33:07 2020

@author: Fazuximy
"""

import pandas as pd
import tkinter as tk
import math
import re
import logging

from tkinter import *
from tkinter.ttk import *

from searching_module import get_interface_results
import document_highlights as dh

from data_constants import OUTPUT_DIR, DATA_DIR, RAW_DATA_NAME, IMAGE_DIR, IMAGE_ARROW_RIGHT, IMAGE_ARROW_LEFT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

input_data = pd.DataFrame(columns=[
    'rank', 'title', 'path', 'document', 'highlight_text', 'last_updated'
])
interface_data = input_data
search_hits = []
tag_index = ""


def filter_all():
    global interface_data
    interface_data = input_data
    text.delete('1.0', tk.END)
    global page_number
    page_number = 0
    show_search_results(page_number)


def filter_business():
    global interface_data
    interface_data = input_data.loc[[("Forretningsgang" in i[0:15])
                                     for i in input_data["path"]]]
    text.delete('1.0', tk.END)
    global page_number
    page_number = 0
    show_search_results(page_number)


def filter_products():
    global interface_data
    interface_data = input_data.loc[[("Produktoversigt" in i[0:15])
                                     for i in input_data["path"]]]
    text.delete('1.0', tk.END)
    global page_number
    page_number = 0
    show_search_results(page_number)


def search_button():
    global input_data
    global interface_data
    global search_hits
    query = e1.get()
    if query != "":
        _root = tk.Tk()
        _root.geometry("+500+100")
        _doc_text = tk.Text(_root, height=10, width=100)
        _doc_text.tag_configure('title', font=('Arial', 20, 'bold'))
        _doc_text.insert(tk.END,
                         "Systemet arbejder på at finde søgeresultater",
                         "title")
        _doc_text.grid(row=0, column=0)
        _root.update()
        input_data, result_hits = get_interface_results(query,
                                                        DATA_DIR +
                                                        RAW_DATA_NAME,
                                                        limit_results=2000)
        _root.destroy()
        text.delete('1.0', tk.END)
        global page_number
        page_number = 0
        interface_data = input_data
        search_hits = result_hits
        show_search_results(page_number)


def enter_title(event):
    index = event.widget.index("@%s,%s" % (event.x, event.y))
    tag = text.tag_names(index=index)[3]
    text.config(cursor="hand2")
    text.tag_configure(tag, font=('Arial', 20), foreground="blue", underline=1)
    return


def click_title(event):

    index = event.widget.index("@%s,%s" % (event.x, event.y))
    tag = text.tag_names(index=index)[3]
    global tag_index
    if tag in title_tags:
        tag_index = title_tags.index(tag)
        create_preview(tag_index)
    return


def open_document():
    if tag_index != "":
        _root = tk.Tk()
        #_root.state("zoomed")

        _empty_text = tk.Text(_root, height=62, width=10, borderwidth=0)
        _doc_text = tk.Text(_root, height=62, width=225, borderwidth=0)
        _scroll = tk.Scrollbar(_root, command=_doc_text.yview)
        _doc_text.configure(yscrollcommand=_scroll.set)
        _doc_text.tag_configure('document', font=('Times', 14), tabs="1c")
        _doc_text.tag_configure('title', font=('Arial', 20, 'bold'))
        _doc_text.tag_configure('path',
                                font=('Arial', 14, 'italic'),
                                foreground="grey")
        _document_title = interface_data.loc[:, 'title'].iloc[tag_index]
        _document_last_updated = interface_data.loc[:, 'last_updated'].iloc[
            tag_index]
        _document_path = interface_data.loc[:, 'path'].iloc[tag_index]
        _document_text = interface_data.loc[:, 'document'].iloc[tag_index]
        _doc_text.insert(tk.END, _document_title + "\n", "title")
        _doc_text.insert(tk.END, _document_path + "\n\n", "path")
        _doc_text.insert(
            tk.END, "Sidst opdateret: " + _document_last_updated + "\n\n\n",
            "path")
        _doc_text.insert(tk.END, _document_text, "document")
        _empty_text.grid(row=0, column=0)
        _doc_text.grid(row=0, column=1)
        _scroll.grid(row=0, column=2, sticky='ns')
        _root.update()
    return


def leave_title(event):
    text.config(cursor="arrow")
    [
        text.tag_configure(i,
                           font=('Arial', 20),
                           foreground="blue",
                           underline=0) for i in title_tags
    ]
    return


def next_page():
    global page_number
    page_number = page_number + 1
    text.delete('1.0', tk.END)
    show_search_results(page_number)


def prev_page():
    global page_number
    page_number = page_number - 1
    text.delete('1.0', tk.END)
    show_search_results(page_number)


def delete_preview():
    text_preview.delete('1.0', tk.END)
    global tag_index
    tag_index = ""
    show_search_results(page_number)
    return


def create_preview(_tag_index):

    text_preview.config(state="normal")
    text_preview.delete('1.0', tk.END)
    _document_title = interface_data.loc[:, 'title'].iloc[_tag_index]
    _document_last_updated = interface_data.loc[:, 'last_updated'].iloc[
        _tag_index]
    _document_path = interface_data.loc[:, 'path'].iloc[_tag_index]
    _document_text = interface_data.loc[:, 'document'].iloc[_tag_index]
    text_preview.insert(tk.END, _document_title + "\n", "title")
    text_preview.insert(tk.END, _document_path + "\n\n", "path")
    text_preview.insert(
        tk.END, "Sidst opdateret: " + _document_last_updated + "\n\n\n",
        "path")
    text_preview.insert(tk.END, _document_text, "document")
    text_preview.config(state="disabled")

    return


def get_position(_pattern_result, _string):
    _p = re.compile(_pattern_result)
    _position = []
    _term = []
    for m in _p.finditer(_string):
        _position.append(m.start())
        _term.append(m.group())
    return (_position, _term)


def tag_highlighted_text(_txt):
    regex_terms_match = r'(?<=<strong class="match term[0-9]">)[^<>]+(?=</strong>)'
    regex_between_match = r'(?<=</strong>)[^<>]+(?=<strong class="match term[0-9]">)'
    regex_start_match = r'^[^<>]+(?=<strong class="match term[0-9]">)'
    regex_end_match = r'(?<=</strong>)[^<>]+$'
    combined_regex = '(' + ')|('.join([
        regex_start_match, regex_terms_match, regex_between_match,
        regex_end_match
    ]) + ')'
    results = re.findall(combined_regex, _txt, re.DOTALL)
    return (results)


page_number = 0

root = tk.Tk()
arrow_right = PhotoImage(file=IMAGE_DIR + IMAGE_ARROW_RIGHT, master=root)
small_right = arrow_right.subsample(10, 10)
arrow_left = PhotoImage(file=IMAGE_DIR + IMAGE_ARROW_LEFT, master=root)
small_left = arrow_left.subsample(10, 10)
#root.attributes('-fullscreen',True)
root.configure(background="white")
#root.state('zoomed')
root.title("The Golden Ad-hoc Retriever")
e1 = tk.Entry(root, width=53, font="Arial 18", borderwidth=2)
text_empty = tk.Text(root, height=51, width=10, borderwidth=0)
search_b = tk.Button(root,
                     text="Søg",
                     command=search_button,
                     font="Arial 18",
                     bg="white",
                     cursor="hand2")

next_button = tk.Button(root,
                        text="Næste side",
                        command=next_page,
                        font="Arial 12",
                        bg="white",
                        overrelief="ridge",
                        cursor="hand2",
                        image=small_right,
                        compound="right")

prev_button = tk.Button(root,
                        text="Forrige side",
                        command=prev_page,
                        font="Arial 12",
                        bg="white",
                        overrelief="ridge",
                        cursor="hand2",
                        image=small_left,
                        compound="left")

delete_button = tk.Button(root,
                          text="Ryd visning",
                          command=delete_preview,
                          font="Arial 12",
                          bg="white",
                          overrelief="ridge",
                          cursor="hand2")

open_button = tk.Button(root,
                        text="Åben dokument",
                        command=open_document,
                        font="Arial 12",
                        bg="white",
                        overrelief="ridge",
                        cursor="hand2")

v = tk.IntVar()
v.set(1)

radio_all = tk.Radiobutton(root,
                           text="Alle dokumenter",
                           variable=v,
                           indicatoron=0,
                           value=1,
                           font="Arial 12",
                           bg="white",
                           selectcolor="light grey",
                           overrelief="ridge",
                           cursor="hand2",
                           command=filter_all)
radio_business = tk.Radiobutton(root,
                                text="Forretningsgange",
                                variable=v,
                                indicatoron=0,
                                value=2,
                                font="Arial 12",
                                bg="white",
                                selectcolor="light grey",
                                overrelief="ridge",
                                cursor="hand2",
                                command=filter_business)
radio_products = tk.Radiobutton(root,
                                text="Produktoversigt",
                                variable=v,
                                indicatoron=0,
                                value=3,
                                font="Arial 12",
                                bg="white",
                                selectcolor="light grey",
                                overrelief="ridge",
                                cursor="hand2",
                                command=filter_products)


def show_search_results(_page_number):

    global interface_data
    _last_page = 0

    global text
    text = tk.Text(root, height=51, width=120, borderwidth=0)
    scroll = tk.Scrollbar(root, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    text.config(cursor="arrow")
    text.tag_configure('path', font=('Arial', 12, 'italic'), foreground="grey")
    text.tag_configure('number', font=('Arial', 12), foreground="grey")
    text.tag_configure('text', font=('Arial', 12))
    text.tag_configure('highlight_text', font=('Arial', 12, 'bold'))
    text.tag_configure('newline', font=('Arial', 4))
    found_results = len(interface_data)

    global text_preview
    text_preview = tk.Text(root, height=45, width=100, borderwidth=0)
    scroll_preview = tk.Scrollbar(root, command=text_preview.yview)
    text_preview.configure(yscrollcommand=scroll_preview.set)
    text_preview.config(cursor="arrow")
    text_preview.tag_configure('path',
                               font=('Arial', 14, 'italic'),
                               foreground="grey")
    text_preview.tag_configure('document', font=('Times', 14), tabs="1c")
    text_preview.tag_configure('title', font=('Arial', 20, 'bold'))
    if (_page_number * 10 + 10) > found_results:
        _numbers_left = found_results - (_page_number * 10)
        _last_page = 1
    else:
        _last_page = 0
        _numbers_left = 10

    text.insert(
        tk.END,
        str(_numbers_left) + " ud af " + str(found_results) +
        " dokumenter\n\n", 'number')

    global title_tags
    title_tags = ['title' + str(i) for i in list(range(0, found_results))]

    for i in list(range(_page_number * 10, _page_number * 10 + _numbers_left)):
        text.tag_bind('enter', '<Enter>', enter_title)
        text.tag_bind('leave', '<Leave>', leave_title)
        text.tag_bind('click', '<Button-1>', click_title)
        text.tag_configure(title_tags[i],
                           font=('Arial', 20),
                           foreground="blue")

        newline = "\n"
        newlines = "\n\n"
        path = interface_data.loc[:, 'path'].iloc[i]
        title = interface_data.loc[:, 'title'].iloc[i]
        if interface_data.isnull().loc[:, 'highlight_text'].iloc[i] == True:
            logger.debug('Inputting highlighted text for row {}'.format(i))
            highlight_text = dh.get_highlight(search_hits[i])

            logger.debug('Saving highlight for document {}: "{}"'.format(
                i, highlight_text))
            interface_data.loc[:, 'highlight_text'].iloc[i] = highlight_text
            logger.debug('Inserted text: {}'.format(
                interface_data.loc[:, 'highlight_text'].iloc[i]))
        else:
            highlight_text = interface_data.loc[:, 'highlight_text'].iloc[i]
        highlight_text_data = tag_highlighted_text(highlight_text)
        dato = interface_data.loc[:, 'last_updated'].iloc[i][:10]

        #text.insert(tk.END, space, 'path')
        text.insert(tk.END, path + "\n", 'path')
        text.insert(tk.END, newline, 'newline')
        text.insert(tk.END, title + "\n",
                    ('title' + str(i), 'enter', 'leave', 'click'))
        text.insert(tk.END, newline, 'newline')
        text.insert(tk.END, dato + " - ", 'path')
        for start, term, between, end in highlight_text_data:
            text.insert(tk.END, start, 'text')
            text.insert(tk.END, term, 'highlight_text')
            text.insert(tk.END, between, 'text')
            text.insert(tk.END, end, 'text')
        #[text.insert(tk.END, highlight_text_data[1][i],
        #             highlight_text_data[2][i]) for i in list(range(0,len(highlight_text_data[0])))]
        text.insert(tk.END, "\n", 'text')
        text.insert(tk.END, newlines, 'path')

    text_empty.config(state="disabled")
    text.config(state="disabled")
    text_preview.config(state="disabled")
    if found_results == 0:
        tk.Label(root,
                 text="Side " + str(_page_number) + " ud af " +
                 str(math.ceil(found_results / 10)),
                 bg="white",
                 font=('Arial', 12)).grid(row=8, column=1, columnspan=3)
    else:
        tk.Label(root,
                 text="Side " + str(_page_number + 1) + " ud af " +
                 str(math.ceil(found_results / 10)),
                 bg="white",
                 font=('Arial', 12)).grid(row=8, column=1, columnspan=3)
    tk.ttk.Separator(root, orient="horizontal").grid(row=3,
                                                     column=0,
                                                     columnspan=20,
                                                     sticky="ew")
    tk.ttk.Separator(root, orient="horizontal").grid(row=7,
                                                     column=0,
                                                     columnspan=20,
                                                     sticky="ew",
                                                     pady=10)
    tk.ttk.Separator(root, orient="horizontal").grid(row=5,
                                                     column=12,
                                                     columnspan=8,
                                                     sticky="ew")

    tk.Label(root,
             text="Visning af dokument",
             bg="white",
             fg="grey",
             font=('Arial', 20)).grid(row=4, column=12, columnspan=8)

    text_empty.grid(row=4, column=0, rowspan=3)
    text.grid(row=4, column=1, columnspan=10, rowspan=3)
    text_preview.grid(row=6, column=12, columnspan=8)
    scroll.grid(row=4, column=11, sticky='ns', rowspan=3, padx=10)
    scroll_preview.grid(row=6, column=20, sticky='ns')
    e1.grid(row=1,
            column=0,
            padx=10,
            pady=30,
            columnspan=20,
            ipady=4,
            ipadx=10)
    search_b.grid(row=1, column=7, columnspan=14)
    delete_button.grid(row=4, column=19)
    open_button.grid(row=4, column=12)

    radio_all.grid(row=2, column=1)
    radio_business.grid(row=2, column=2)
    radio_products.grid(row=2, column=3)

    if _last_page == 1:
        next_button.grid_forget()
    else:
        next_button.grid(row=8, column=3)

    if page_number <= 0:
        prev_button.grid_forget()
    else:
        prev_button.grid(row=8, column=1)

    root.grid_columnconfigure(4, minsize=100)

    tk.mainloop()


show_search_results(page_number)
