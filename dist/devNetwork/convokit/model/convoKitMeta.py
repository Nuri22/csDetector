from collections import MutableMapping
from convokit.util import warn
from .convoKitIndex import ConvoKitIndex
import json

# See reference: https://stackoverflow.com/questions/7760916/correct-usage-of-a-getter-setter-for-dictionary-values


class ConvoKitMeta(MutableMapping, dict):
    """
    ConvoKitMeta is a dictlike object that stores the metadata attributes of a corpus component
    """
    def __init__(self, convokit_index, obj_type):
        self.index: ConvoKitIndex = convokit_index
        self.obj_type = obj_type

    def __getitem__(self, item):
        return dict.__getitem__(self, item)

    @staticmethod
    def _check_type_and_update_index(index, obj_type, key, value):
        if not isinstance(value, type(None)): # do nothing to index if value is None
            if key not in index.indices[obj_type]:
                type_ = _optimized_type_check(value)
                index.update_index(obj_type, key=key, class_type=type_)
            else:
                # entry exists
                if index.get_index(obj_type)[key] != ["bin"]: # if "bin" do no further checks
                    if str(type(value)) not in index.get_index(obj_type)[key]:
                        new_type = _optimized_type_check(value)

                        if new_type == "bin":
                            index.set_index(obj_type, key, "bin")
                        else:
                            index.update_index(obj_type, key, new_type)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            warn("Metadata attribute keys must be strings. Input key has been casted to a string.")
            key = str(key)

        if self.index.type_check:
            ConvoKitMeta._check_type_and_update_index(self.index, self.obj_type, key, value)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        if self.obj_type == 'corpus':
            dict.__delitem__(self, key)
            self.index.del_from_index(self.obj_type, key)
        else:
            if self.index.lock_metadata_deletion[self.obj_type]:
                warn("For consistency in metadata attributes in Corpus component objects, deleting metadata attributes "
                         "from component objects individually is not allowed. "
                     "To delete this metadata attribute from all Corpus components of this type, "
                         "use corpus.delete_metadata(obj_type='{}', attribute='{}') instead.".format(self.obj_type, key))
            else:
                dict.__delitem__(self, key)

    def __iter__(self):
        return dict.__iter__(self)

    def __len__(self):
        return dict.__len__(self)

    def __contains__(self, x):
        return dict.__contains__(self, x)

    def to_dict(self):
        return self.__dict__


_basic_types = {type(0), type(1.0), type('str'), type(True)} # cannot include lists or dicts


def _optimized_type_check(val):
    # if type(obj)
    if type(val) in _basic_types:
        return str(type(val))
    else:
        try:
            json.dumps(val)
            return str(type(val))
        except (TypeError, OverflowError):
            return "bin"
