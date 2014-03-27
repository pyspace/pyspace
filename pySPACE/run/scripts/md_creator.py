""" Create meta data file 'metadata.yaml' for :class:`~pySPACE.resources.dataset_defs.feature_vector.FeatureVectorCollection`
 
Used for external files, which can not be read directly in pySPACE.
Eg. csv files without names.
To be called in the dataset directory.
"""

def main(md_file):
    # Request all necessary data from user
    data={}
    msg = "Please enter the name of the file. --> "
    data['file_name'] = get_user_input(msg)
    
    msg = "Please enter the storage_format of the data.\n "
    msg += "one of arff, csv (csv with header), csvUnnamed (csv without header)--> "
    data['format'] = get_user_input(msg)
    
    if data['format'] != 'arff':
        while True:
            msg  = "Please enter csv delimiter/separator. \n"
            msg += "(e.g. ','  , ' ' , ';' or '\t' for a tab, default:',')-->"
            data['delimiter'] = get_user_input(msg)
            if check_delimiter(data):
                break
        
        msg = "Please enter all rows that can be ignored, separated by comma or range.\n "
        msg += "eg. [1, 2, 3] or [1-3] --> "
        data['rows'] = get_numerical_user_input(msg)
      
        msg = "Please enter all columns that can be ignored, separated by comma or range.\n"
        msg += "The first row gets number 1."
        msg += "eg. [1, 2, 3] or [1-3] --> "
        data['columns'] = get_numerical_user_input(msg)
    
        msg = 'Please enter the column that contains the label. -1 for last column \n --> '
        data['label'] = get_user_input(msg)
    
    meta_data = generate_meta_data(data)
    
    write_md_file(meta_data, md_file)
            
    info_string = """\nMeta data file %s written. \n"""%md_file
    give_info(info_string)

def get_numerical_user_input(msg):
    """ Request input, split it by ',' and parse it for '-' """
    tmp_info = raw_input(msg)
    tmp_info = tmp_info.replace(' ', '').split(',')
    return parse_list(tmp_info)

def get_user_input(msg):
    """ Request input """
    return raw_input(msg)


def parse_list(input_list):
    """ Replace range by explicit numbers """
    info = []
    for index in input_list:
        if type(index) == int:
            info.append(index)
        if not type(index) == str:
            info.append(int(index))
        # zero is not an accepted index
        if index == '0' or index == '':
            continue
        # replacing '-' with actual indices
        if '-' in str(index):
            index_split = index.split('-')
            # to handle -1 input
            if index_split[0] == '':
                info.append(int(index))
                continue
            low = int(index_split[0])
            high = int(index_split[1])
            rnge = high - low
            new_index = [low]
            for i in range(rnge):
                new_index.append(low + i + 1)
            info = info.extend(new_index)
        else:
            info.append(int(index))

    return info

def check_delimiter(data):
    """ Checks delimiter to have length one """
    delimiter = data["delimiter"]
    if len(delimiter) == 0:
        # add the deleted spaces
        data["delimiter"]=' '
        return True
    elif len(delimiter)==1:
        # tabulator is included here
        return True
    else:
        import warnings
        warnings.warn('To long delimiter. Only 1 sign allowed. Please try again.')

def generate_meta_data(data):
    """ Map data to the metadata.yaml string and set defaults """
    meta_data = "author: " + os.environ['USER'] + '\n' + \
        "date: " + time.strftime("%Y%m%d")+ '\n' + \
        "type: feature_vector" + "\n"
    for item in data.items():
        if item[1] != '':
            if item[0] == 'file_name':
                meta_data += "file_name: " + str(data["file_name"]) + "\n"
            elif item[0] == 'format':
                meta_data += "storage_format: [" + str(data["format"]) + ', real]' + "\n"
            elif item[0] == 'rows':
                meta_data += "ignored_rows: " + str(data["rows"]) + "\n"
            elif item[0] == 'columns':
                meta_data += "ignored_columns: " + str(data["columns"]) + "\n"
            elif item[0] == 'label':
                meta_data += "label_column: " + str(data["label"]) + "\n"
        else: # set defaults
            if item[0] == 'file_name':
                meta_data += "file_name: " + "file_name.csv" + "\n"
            elif item[0] == 'format':
                meta_data += "storage_format: [" + "csv" + ', real]' + "\n"
            elif item[0] == 'rows':
                meta_data += "ignored_rows: " + "[]" + "\n"
            elif item[0] == 'columns':
                meta_data += "ignored_columns: " + "[]" + "\n"
            elif item[0] == 'label':
                meta_data += "label_column: " + str(-1) + "\n"
    return meta_data

    
def write_md_file(meta_data, md_file):
    
    meta_data_file = open(md_file, "w")
    meta_data_file.write(meta_data)
    meta_data_file.close()
    
    
def give_info(msg):
    
    print msg


import os, time, sys

if __name__ == "__main__":

    info_string = "\nRunning meta data creator ... \n"
    give_info(info_string)
    
    md_file = "metadata.yaml"
    
    if not os.path.isfile(md_file):
        main(md_file)
    else:
        msg = "'metadata.yaml' already exists! \n"
        give_info(msg)
        yes_no = raw_input("Overwrite? y/n: ")
        if yes_no == "y":
            main(md_file)
        else:
            msg = "Exiting ... \n"
            give_info(msg)
            sys.exit(0)
