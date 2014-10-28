""" This script renames the collection names to intra-run, intra-session, 
inter-session, inter-subject.

This can be done only if the original
#collection names follow the following naming convention:
subject1name_session1name_run1name_vs_subject2name_session2name_run2name
The underscores and the "_vs_" are used to parse the name. If e.g.
subject1name !=  subject2name, the inter-subject case is identified
"""
if __name__ == '__main__':
    import csv
    from collections import defaultdict
    
    source_file = open("results.csv", "r")
    source_reader = csv.DictReader(source_file)
    sink_file = open("results_pp.csv", "w")
    sink_writer = None
    
    for line_dict in source_reader:
        tokens = line_dict["__Dataset__"].split('_vs_')
        subject1, session1, run1 = tokens[0].split('_')
        subject2, session2, run2 = tokens[1].split('_')
        if subject1 == subject2 and session1 == session2 and run1 == run2:
            line_dict["__Dataset__"] = "Intra-run"
        elif subject1 == subject2 and session1 == session2:
            assert run1 != runs2
            line_dict["__Dataset__"] = "Intra-session"
        elif subject1 == subject2:
            assert session1 != session2
            line_dict["__Dataset__"] = "Intra-subject"
        else:
            assert subject1 != subject2
            line_dict["__Dataset__"] = "Inter-subject"
        # Lazy initialization
        if sink_writer == None:
            sink_writer = csv.DictWriter(sink_file, line_dict.keys(), lineterminator='\n')
            sink_writer.writerow(dict(zip(line_dict.keys(), line_dict.keys())))
        sink_writer.writerow(line_dict)
        
    sink_file.close()