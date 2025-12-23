# This file was generated from metar_parser.peg
# See https://canopy.jcoglan.com/ for documentation

from collections import defaultdict
import re


class TreeNode(object):
    def __init__(self, text, offset, elements):
        self.text = text
        self.offset = offset
        self.elements = elements

    def __iter__(self):
        for el in self.elements:
            yield el


class TreeNode1(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode1, self).__init__(text, offset, elements)
        self.metar = elements[0]
        self.siteid = elements[1]
        self.datetime = elements[2]
        self.auto = elements[3]
        self.wind = elements[4]
        self.vis = elements[5]
        self.run = elements[6]
        self.curwx = elements[7]
        self.skyc = elements[8]
        self.temp_dewp = elements[9]
        self.altim = elements[10]
        self.remarks = elements[11]
        self.end = elements[12]


class TreeNode2(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode2, self).__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode3(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode3, self).__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode4(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode4, self).__init__(text, offset, elements)
        self.wind_dir = elements[1]
        self.wind_spd = elements[2]
        self.gust = elements[3]


class TreeNode5(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode5, self).__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode6(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode6, self).__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode7(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode7, self).__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode8(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode8, self).__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode9(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode9, self).__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode10(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode10, self).__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode11(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode11, self).__init__(text, offset, elements)
        self.sep = elements[0]
        self.wx = elements[1]


class TreeNode12(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode12, self).__init__(text, offset, elements)
        self.sep = elements[0]
        self.cover = elements[1]


class TreeNode13(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode13, self).__init__(text, offset, elements)
        self.sep = elements[0]
        self.temp = elements[2]
        self.dewp = elements[4]


FAILURE = object()


class Grammar(object):
    REGEX_1 = re.compile('^[0-9A-Z]')
    REGEX_2 = re.compile('^[0-9A-Z]')
    REGEX_3 = re.compile('^[0-9A-Z]')
    REGEX_4 = re.compile('^[0-9A-Z]')
    REGEX_5 = re.compile('^[\\d]')
    REGEX_6 = re.compile('^[\\d]')
    REGEX_7 = re.compile('^[\\d]')
    REGEX_8 = re.compile('^[\\d]')
    REGEX_9 = re.compile('^[\\d]')
    REGEX_10 = re.compile('^[\\d]')
    REGEX_11 = re.compile('^[\\d]')
    REGEX_12 = re.compile('^[\\d]')
    REGEX_13 = re.compile('^[\\d]')
    REGEX_14 = re.compile('^[\\d]')
    REGEX_15 = re.compile('^[\\d]')
    REGEX_16 = re.compile('^[\\d]')
    REGEX_17 = re.compile('^[\\d]')
    REGEX_18 = re.compile('^[\\d]')
    REGEX_19 = re.compile('^[\\d]')
    REGEX_20 = re.compile('^[\\d]')
    REGEX_21 = re.compile('^[\\d]')
    REGEX_22 = re.compile('^[\\d]')
    REGEX_23 = re.compile('^[\\d]')
    REGEX_24 = re.compile('^[\\d]')
    REGEX_25 = re.compile('^[\\d]')
    REGEX_26 = re.compile('^[\\d]')
    REGEX_27 = re.compile('^[\\d]')
    REGEX_28 = re.compile('^[\\d]')
    REGEX_29 = re.compile('^[\\d]')
    REGEX_30 = re.compile('^[\\d]')
    REGEX_31 = re.compile('^[\\d]')
    REGEX_32 = re.compile('^[\\d]')
    REGEX_33 = re.compile('^[\\d]')
    REGEX_34 = re.compile('^[NSEW]')
    REGEX_35 = re.compile('^[NSEW]')
    REGEX_36 = re.compile('^[LRC]')
    REGEX_37 = re.compile('^[\\d]')
    REGEX_38 = re.compile('^[\\d]')
    REGEX_39 = re.compile('^[LRC]')
    REGEX_40 = re.compile('^[\\d]')
    REGEX_41 = re.compile('^[\\d]')
    REGEX_42 = re.compile('^[\\d]')
    REGEX_43 = re.compile('^[\\d]')
    REGEX_44 = re.compile('^["M" / "P"]')
    REGEX_45 = re.compile('^[\\d]')
    REGEX_46 = re.compile('^[\\d]')
    REGEX_47 = re.compile('^[\\d]')
    REGEX_48 = re.compile('^[\\d]')
    REGEX_49 = re.compile('^[UDN]')
    REGEX_50 = re.compile('^[-+]')
    REGEX_51 = re.compile('^[\\d]')
    REGEX_52 = re.compile('^[M]')
    REGEX_53 = re.compile('^[\\d]')
    REGEX_54 = re.compile('^[\\d]')
    REGEX_55 = re.compile('^[M]')
    REGEX_56 = re.compile('^[\\d]')
    REGEX_57 = re.compile('^[\\d]')
    REGEX_58 = re.compile('^["Q" / "A"]')
    REGEX_59 = re.compile('^[\\d]')
    REGEX_60 = re.compile('^[\\d]')
    REGEX_61 = re.compile('^[\\d]')
    REGEX_62 = re.compile('^[\\d]')

    def _read_ob(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ob'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_metar()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            address2 = self._read_siteid()
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                address3 = self._read_datetime()
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    address4 = self._read_auto()
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        address5 = self._read_wind()
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            address6 = self._read_vis()
                            if address6 is not FAILURE:
                                elements0.append(address6)
                                address7 = FAILURE
                                address7 = self._read_run()
                                if address7 is not FAILURE:
                                    elements0.append(address7)
                                    address8 = FAILURE
                                    address8 = self._read_curwx()
                                    if address8 is not FAILURE:
                                        elements0.append(address8)
                                        address9 = FAILURE
                                        address9 = self._read_skyc()
                                        if address9 is not FAILURE:
                                            elements0.append(address9)
                                            address10 = FAILURE
                                            address10 = self._read_temp_dewp()
                                            if address10 is not FAILURE:
                                                elements0.append(address10)
                                                address11 = FAILURE
                                                address11 = self._read_altim()
                                                if address11 is not FAILURE:
                                                    elements0.append(address11)
                                                    address12 = FAILURE
                                                    address12 = self._read_remarks()
                                                    if address12 is not FAILURE:
                                                        elements0.append(address12)
                                                        address13 = FAILURE
                                                        address13 = self._read_end()
                                                        if address13 is not FAILURE:
                                                            elements0.append(address13)
                                                        else:
                                                            elements0 = None
                                                            self._offset = index1
                                                    else:
                                                        elements0 = None
                                                        self._offset = index1
                                                else:
                                                    elements0 = None
                                                    self._offset = index1
                                            else:
                                                elements0 = None
                                                self._offset = index1
                                        else:
                                            elements0 = None
                                            self._offset = index1
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode1(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['ob'][index0] = (address0, self._offset)
        return address0

    def _read_metar(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['metar'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0, max0 = None, self._offset + 4
        if max0 <= self._input_size:
            chunk0 = self._input[self._offset:max0]
        if chunk0 == 'COR ':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset, [])
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append(('METAR::metar', '"COR "'))
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2, [])
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            index4 = self._offset
            chunk1, max1 = None, self._offset + 5
            if max1 <= self._input_size:
                chunk1 = self._input[self._offset:max1]
            if chunk1 == 'METAR':
                address2 = TreeNode(self._input[self._offset:self._offset + 5], self._offset, [])
                self._offset = self._offset + 5
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::metar', '"METAR"'))
            if address2 is FAILURE:
                self._offset = index4
                chunk2, max2 = None, self._offset + 5
                if max2 <= self._input_size:
                    chunk2 = self._input[self._offset:max2]
                if chunk2 == 'SPECI':
                    address2 = TreeNode(self._input[self._offset:self._offset + 5], self._offset, [])
                    self._offset = self._offset + 5
                else:
                    address2 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::metar', '"SPECI"'))
                if address2 is FAILURE:
                    self._offset = index4
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index3:index3], index3, [])
                self._offset = index3
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index5 = self._offset
                address3 = self._read_auto()
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index5:index5], index5, [])
                    self._offset = index5
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['metar'][index0] = (address0, self._offset)
        return address0

    def _read_sep(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['sep'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0, address1 = self._offset, [], None
        while True:
            chunk0, max0 = None, self._offset + 1
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 == ' ':
                address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::sep', '" "'))
            if address1 is not FAILURE:
                elements0.append(address1)
            else:
                break
        if len(elements0) >= 1:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        else:
            address0 = FAILURE
        self._cache['sep'][index0] = (address0, self._offset)
        return address0

    def _read_siteid(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['siteid'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2, [])
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0, max0 = None, self._offset + 1
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 is not None and Grammar.REGEX_1.search(chunk0):
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::siteid', '[0-9A-Z]'))
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk1, max1 = None, self._offset + 1
                if max1 <= self._input_size:
                    chunk1 = self._input[self._offset:max1]
                if chunk1 is not None and Grammar.REGEX_2.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::siteid', '[0-9A-Z]'))
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk2, max2 = None, self._offset + 1
                    if max2 <= self._input_size:
                        chunk2 = self._input[self._offset:max2]
                    if chunk2 is not None and Grammar.REGEX_3.search(chunk2):
                        address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::siteid', '[0-9A-Z]'))
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        chunk3, max3 = None, self._offset + 1
                        if max3 <= self._input_size:
                            chunk3 = self._input[self._offset:max3]
                        if chunk3 is not None and Grammar.REGEX_4.search(chunk3):
                            address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::siteid', '[0-9A-Z]'))
                        if address5 is not FAILURE:
                            elements0.append(address5)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['siteid'][index0] = (address0, self._offset)
        return address0

    def _read_datetime(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['datetime'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2, elements1, address3 = self._offset, [], None
            while True:
                chunk0, max0 = None, self._offset + 1
                if max0 <= self._input_size:
                    chunk0 = self._input[self._offset:max0]
                if chunk0 is not None and Grammar.REGEX_5.search(chunk0):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::datetime', '[\\d]'))
                if address3 is not FAILURE:
                    elements1.append(address3)
                else:
                    break
            if len(elements1) >= 1:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                chunk1, max1 = None, self._offset + 1
                if max1 <= self._input_size:
                    chunk1 = self._input[self._offset:max1]
                if chunk1 == 'Z':
                    address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::datetime', '"Z"'))
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode2(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['datetime'][index0] = (address0, self._offset)
        return address0

    def _read_auto(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['auto'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0, address1 = self._offset, [], None
        while True:
            index3, elements1 = self._offset, []
            address2 = FAILURE
            address2 = self._read_sep()
            if address2 is not FAILURE:
                elements1.append(address2)
                address3 = FAILURE
                index4 = self._offset
                chunk0, max0 = None, self._offset + 4
                if max0 <= self._input_size:
                    chunk0 = self._input[self._offset:max0]
                if chunk0 == 'AUTO':
                    address3 = TreeNode(self._input[self._offset:self._offset + 4], self._offset, [])
                    self._offset = self._offset + 4
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::auto', '"AUTO"'))
                if address3 is FAILURE:
                    self._offset = index4
                    chunk1, max1 = None, self._offset + 3
                    if max1 <= self._input_size:
                        chunk1 = self._input[self._offset:max1]
                    if chunk1 == 'COR':
                        address3 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                        self._offset = self._offset + 3
                    else:
                        address3 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::auto', '"COR"'))
                    if address3 is FAILURE:
                        self._offset = index4
                if address3 is not FAILURE:
                    elements1.append(address3)
                else:
                    elements1 = None
                    self._offset = index3
            else:
                elements1 = None
                self._offset = index3
            if elements1 is None:
                address1 = FAILURE
            else:
                address1 = TreeNode3(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            if address1 is not FAILURE:
                elements0.append(address1)
            else:
                break
        if len(elements0) >= 1:
            address0 = TreeNode(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        else:
            address0 = FAILURE
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['auto'][index0] = (address0, self._offset)
        return address0

    def _read_wind(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['wind'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index3:index3], index3, [])
            self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            address2 = self._read_wind_dir()
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                address3 = self._read_wind_spd()
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    address4 = self._read_gust()
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        index4 = self._offset
                        chunk0, max0 = None, self._offset + 2
                        if max0 <= self._input_size:
                            chunk0 = self._input[self._offset:max0]
                        if chunk0 == 'KT':
                            address5 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                            self._offset = self._offset + 2
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::wind', '"KT"'))
                        if address5 is FAILURE:
                            self._offset = index4
                            chunk1, max1 = None, self._offset + 3
                            if max1 <= self._input_size:
                                chunk1 = self._input[self._offset:max1]
                            if chunk1 == 'MPS':
                                address5 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                                self._offset = self._offset + 3
                            else:
                                address5 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::wind', '"MPS"'))
                            if address5 is FAILURE:
                                self._offset = index4
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            index5 = self._offset
                            address6 = self._read_varwind()
                            if address6 is FAILURE:
                                address6 = TreeNode(self._input[index5:index5], index5, [])
                                self._offset = index5
                            if address6 is not FAILURE:
                                elements0.append(address6)
                            else:
                                elements0 = None
                                self._offset = index2
                        else:
                            elements0 = None
                            self._offset = index2
                    else:
                        elements0 = None
                        self._offset = index2
                else:
                    elements0 = None
                    self._offset = index2
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode4(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['wind'][index0] = (address0, self._offset)
        return address0

    def _read_wind_dir(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['wind_dir'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2 = self._offset
        index3, elements0 = self._offset, []
        address1 = FAILURE
        chunk0, max0 = None, self._offset + 1
        if max0 <= self._input_size:
            chunk0 = self._input[self._offset:max0]
        if chunk0 is not None and Grammar.REGEX_6.search(chunk0):
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append(('METAR::wind_dir', '[\\d]'))
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk1, max1 = None, self._offset + 1
            if max1 <= self._input_size:
                chunk1 = self._input[self._offset:max1]
            if chunk1 is not None and Grammar.REGEX_7.search(chunk1):
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::wind_dir', '[\\d]'))
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk2, max2 = None, self._offset + 1
                if max2 <= self._input_size:
                    chunk2 = self._input[self._offset:max2]
                if chunk2 is not None and Grammar.REGEX_8.search(chunk2):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::wind_dir', '[\\d]'))
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index3
            else:
                elements0 = None
                self._offset = index3
        else:
            elements0 = None
            self._offset = index3
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index3:self._offset], index3, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index2
            chunk3, max3 = None, self._offset + 3
            if max3 <= self._input_size:
                chunk3 = self._input[self._offset:max3]
            if chunk3 == 'VAR':
                address0 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                self._offset = self._offset + 3
            else:
                address0 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::wind_dir', '\'VAR\''))
            if address0 is FAILURE:
                self._offset = index2
                chunk4, max4 = None, self._offset + 3
                if max4 <= self._input_size:
                    chunk4 = self._input[self._offset:max4]
                if chunk4 == 'VRB':
                    address0 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                    self._offset = self._offset + 3
                else:
                    address0 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::wind_dir', '\'VRB\''))
                if address0 is FAILURE:
                    self._offset = index2
                    chunk5, max5 = None, self._offset + 3
                    if max5 <= self._input_size:
                        chunk5 = self._input[self._offset:max5]
                    if chunk5 == '///':
                        address0 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                        self._offset = self._offset + 3
                    else:
                        address0 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::wind_dir', '"///"'))
                    if address0 is FAILURE:
                        self._offset = index2
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['wind_dir'][index0] = (address0, self._offset)
        return address0

    def _read_wind_spd(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['wind_spd'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2 = self._offset
        index3, elements0 = self._offset, []
        address1 = FAILURE
        chunk0, max0 = None, self._offset + 1
        if max0 <= self._input_size:
            chunk0 = self._input[self._offset:max0]
        if chunk0 is not None and Grammar.REGEX_9.search(chunk0):
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append(('METAR::wind_spd', '[\\d]'))
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk1, max1 = None, self._offset + 1
            if max1 <= self._input_size:
                chunk1 = self._input[self._offset:max1]
            if chunk1 is not None and Grammar.REGEX_10.search(chunk1):
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::wind_spd', '[\\d]'))
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index4 = self._offset
                chunk2, max2 = None, self._offset + 1
                if max2 <= self._input_size:
                    chunk2 = self._input[self._offset:max2]
                if chunk2 is not None and Grammar.REGEX_11.search(chunk2):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::wind_spd', '[\\d]'))
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index4:index4], index4, [])
                    self._offset = index4
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index3
            else:
                elements0 = None
                self._offset = index3
        else:
            elements0 = None
            self._offset = index3
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index3:self._offset], index3, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index2
            chunk3, max3 = None, self._offset + 2
            if max3 <= self._input_size:
                chunk3 = self._input[self._offset:max3]
            if chunk3 == '//':
                address0 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                self._offset = self._offset + 2
            else:
                address0 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::wind_spd', '"//"'))
            if address0 is FAILURE:
                self._offset = index2
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['wind_spd'][index0] = (address0, self._offset)
        return address0

    def _read_gust(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['gust'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        chunk0, max0 = None, self._offset + 1
        if max0 <= self._input_size:
            chunk0 = self._input[self._offset:max0]
        if chunk0 == 'G':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append(('METAR::gust', '"G"'))
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3, elements1, address3 = self._offset, [], None
            while True:
                chunk1, max1 = None, self._offset + 1
                if max1 <= self._input_size:
                    chunk1 = self._input[self._offset:max1]
                if chunk1 is not None and Grammar.REGEX_12.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::gust', '[\\d]'))
                if address3 is not FAILURE:
                    elements1.append(address3)
                else:
                    break
            if len(elements1) >= 1:
                address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['gust'][index0] = (address0, self._offset)
        return address0

    def _read_varwind(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['varwind'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0, max0 = None, self._offset + 1
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 is not None and Grammar.REGEX_13.search(chunk0):
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::varwind', '[\\d]'))
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk1, max1 = None, self._offset + 1
                if max1 <= self._input_size:
                    chunk1 = self._input[self._offset:max1]
                if chunk1 is not None and Grammar.REGEX_14.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::varwind', '[\\d]'))
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk2, max2 = None, self._offset + 1
                    if max2 <= self._input_size:
                        chunk2 = self._input[self._offset:max2]
                    if chunk2 is not None and Grammar.REGEX_15.search(chunk2):
                        address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::varwind', '[\\d]'))
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        chunk3, max3 = None, self._offset + 1
                        if max3 <= self._input_size:
                            chunk3 = self._input[self._offset:max3]
                        if chunk3 == 'V':
                            address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::varwind', '"V"'))
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            chunk4, max4 = None, self._offset + 1
                            if max4 <= self._input_size:
                                chunk4 = self._input[self._offset:max4]
                            if chunk4 is not None and Grammar.REGEX_16.search(chunk4):
                                address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                self._offset = self._offset + 1
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::varwind', '[\\d]'))
                            if address6 is not FAILURE:
                                elements0.append(address6)
                                address7 = FAILURE
                                chunk5, max5 = None, self._offset + 1
                                if max5 <= self._input_size:
                                    chunk5 = self._input[self._offset:max5]
                                if chunk5 is not None and Grammar.REGEX_17.search(chunk5):
                                    address7 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                    self._offset = self._offset + 1
                                else:
                                    address7 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append(('METAR::varwind', '[\\d]'))
                                if address7 is not FAILURE:
                                    elements0.append(address7)
                                    address8 = FAILURE
                                    chunk6, max6 = None, self._offset + 1
                                    if max6 <= self._input_size:
                                        chunk6 = self._input[self._offset:max6]
                                    if chunk6 is not None and Grammar.REGEX_18.search(chunk6):
                                        address8 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                        self._offset = self._offset + 1
                                    else:
                                        address8 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append(('METAR::varwind', '[\\d]'))
                                    if address8 is not FAILURE:
                                        elements0.append(address8)
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode5(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['varwind'][index0] = (address0, self._offset)
        return address0

    def _read_vis(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['vis'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            index4, elements1 = self._offset, []
            address3 = FAILURE
            chunk0, max0 = None, self._offset + 1
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 is not None and Grammar.REGEX_19.search(chunk0):
                address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address3 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::vis', '[\\d]'))
            if address3 is not FAILURE:
                elements1.append(address3)
                address4 = FAILURE
                chunk1, max1 = None, self._offset + 1
                if max1 <= self._input_size:
                    chunk1 = self._input[self._offset:max1]
                if chunk1 is not None and Grammar.REGEX_20.search(chunk1):
                    address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::vis', '[\\d]'))
                if address4 is not FAILURE:
                    elements1.append(address4)
                    address5 = FAILURE
                    chunk2, max2 = None, self._offset + 1
                    if max2 <= self._input_size:
                        chunk2 = self._input[self._offset:max2]
                    if chunk2 is not None and Grammar.REGEX_21.search(chunk2):
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::vis', '[\\d]'))
                    if address5 is not FAILURE:
                        elements1.append(address5)
                        address6 = FAILURE
                        chunk3, max3 = None, self._offset + 1
                        if max3 <= self._input_size:
                            chunk3 = self._input[self._offset:max3]
                        if chunk3 is not None and Grammar.REGEX_22.search(chunk3):
                            address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                            self._offset = self._offset + 1
                        else:
                            address6 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::vis', '[\\d]'))
                        if address6 is not FAILURE:
                            elements1.append(address6)
                            address7 = FAILURE
                            index5 = self._offset
                            chunk4, max4 = None, self._offset + 3
                            if max4 <= self._input_size:
                                chunk4 = self._input[self._offset:max4]
                            if chunk4 == 'NDV':
                                address7 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                                self._offset = self._offset + 3
                            else:
                                address7 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::vis', '"NDV"'))
                            if address7 is FAILURE:
                                address7 = TreeNode(self._input[index5:index5], index5, [])
                                self._offset = index5
                            if address7 is not FAILURE:
                                elements1.append(address7)
                            else:
                                elements1 = None
                                self._offset = index4
                        else:
                            elements1 = None
                            self._offset = index4
                    else:
                        elements1 = None
                        self._offset = index4
                else:
                    elements1 = None
                    self._offset = index4
            else:
                elements1 = None
                self._offset = index4
            if elements1 is None:
                address2 = FAILURE
            else:
                address2 = TreeNode(self._input[index4:self._offset], index4, elements1)
                self._offset = self._offset
            if address2 is FAILURE:
                self._offset = index3
                index6, elements2 = self._offset, []
                address8 = FAILURE
                chunk5, max5 = None, self._offset + 1
                if max5 <= self._input_size:
                    chunk5 = self._input[self._offset:max5]
                if chunk5 is not None and Grammar.REGEX_23.search(chunk5):
                    address8 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address8 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::vis', '[\\d]'))
                if address8 is not FAILURE:
                    elements2.append(address8)
                    address9 = FAILURE
                    index7 = self._offset
                    index8 = self._offset
                    chunk6, max6 = None, self._offset + 1
                    if max6 <= self._input_size:
                        chunk6 = self._input[self._offset:max6]
                    if chunk6 is not None and Grammar.REGEX_24.search(chunk6):
                        address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address9 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::vis', '[\\d]'))
                    if address9 is FAILURE:
                        self._offset = index8
                        index9, elements3 = self._offset, []
                        address10 = FAILURE
                        index10 = self._offset
                        index11, elements4 = self._offset, []
                        address11 = FAILURE
                        chunk7, max7 = None, self._offset + 1
                        if max7 <= self._input_size:
                            chunk7 = self._input[self._offset:max7]
                        if chunk7 == ' ':
                            address11 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                            self._offset = self._offset + 1
                        else:
                            address11 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::vis', '" "'))
                        if address11 is not FAILURE:
                            elements4.append(address11)
                            address12 = FAILURE
                            chunk8, max8 = None, self._offset + 1
                            if max8 <= self._input_size:
                                chunk8 = self._input[self._offset:max8]
                            if chunk8 is not None and Grammar.REGEX_25.search(chunk8):
                                address12 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                self._offset = self._offset + 1
                            else:
                                address12 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::vis', '[\\d]'))
                            if address12 is not FAILURE:
                                elements4.append(address12)
                            else:
                                elements4 = None
                                self._offset = index11
                        else:
                            elements4 = None
                            self._offset = index11
                        if elements4 is None:
                            address10 = FAILURE
                        else:
                            address10 = TreeNode(self._input[index11:self._offset], index11, elements4)
                            self._offset = self._offset
                        if address10 is FAILURE:
                            address10 = TreeNode(self._input[index10:index10], index10, [])
                            self._offset = index10
                        if address10 is not FAILURE:
                            elements3.append(address10)
                            address13 = FAILURE
                            chunk9, max9 = None, self._offset + 1
                            if max9 <= self._input_size:
                                chunk9 = self._input[self._offset:max9]
                            if chunk9 == '/':
                                address13 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                self._offset = self._offset + 1
                            else:
                                address13 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::vis', '"/"'))
                            if address13 is not FAILURE:
                                elements3.append(address13)
                                address14 = FAILURE
                                chunk10, max10 = None, self._offset + 1
                                if max10 <= self._input_size:
                                    chunk10 = self._input[self._offset:max10]
                                if chunk10 is not None and Grammar.REGEX_26.search(chunk10):
                                    address14 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                    self._offset = self._offset + 1
                                else:
                                    address14 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append(('METAR::vis', '[\\d]'))
                                if address14 is not FAILURE:
                                    elements3.append(address14)
                                    address15 = FAILURE
                                    index12 = self._offset
                                    chunk11, max11 = None, self._offset + 1
                                    if max11 <= self._input_size:
                                        chunk11 = self._input[self._offset:max11]
                                    if chunk11 is not None and Grammar.REGEX_27.search(chunk11):
                                        address15 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                        self._offset = self._offset + 1
                                    else:
                                        address15 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append(('METAR::vis', '[\\d]'))
                                    if address15 is FAILURE:
                                        address15 = TreeNode(self._input[index12:index12], index12, [])
                                        self._offset = index12
                                    if address15 is not FAILURE:
                                        elements3.append(address15)
                                    else:
                                        elements3 = None
                                        self._offset = index9
                                else:
                                    elements3 = None
                                    self._offset = index9
                            else:
                                elements3 = None
                                self._offset = index9
                        else:
                            elements3 = None
                            self._offset = index9
                        if elements3 is None:
                            address9 = FAILURE
                        else:
                            address9 = TreeNode(self._input[index9:self._offset], index9, elements3)
                            self._offset = self._offset
                        if address9 is FAILURE:
                            self._offset = index8
                    if address9 is FAILURE:
                        address9 = TreeNode(self._input[index7:index7], index7, [])
                        self._offset = index7
                    if address9 is not FAILURE:
                        elements2.append(address9)
                        address16 = FAILURE
                        chunk12, max12 = None, self._offset + 2
                        if max12 <= self._input_size:
                            chunk12 = self._input[self._offset:max12]
                        if chunk12 == 'SM':
                            address16 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                            self._offset = self._offset + 2
                        else:
                            address16 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::vis', '"SM"'))
                        if address16 is not FAILURE:
                            elements2.append(address16)
                        else:
                            elements2 = None
                            self._offset = index6
                    else:
                        elements2 = None
                        self._offset = index6
                else:
                    elements2 = None
                    self._offset = index6
                if elements2 is None:
                    address2 = FAILURE
                else:
                    address2 = TreeNode(self._input[index6:self._offset], index6, elements2)
                    self._offset = self._offset
                if address2 is FAILURE:
                    self._offset = index3
                    index13, elements5 = self._offset, []
                    address17 = FAILURE
                    chunk13, max13 = None, self._offset + 1
                    if max13 <= self._input_size:
                        chunk13 = self._input[self._offset:max13]
                    if chunk13 == 'M':
                        address17 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address17 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::vis', '"M"'))
                    if address17 is not FAILURE:
                        elements5.append(address17)
                        address18 = FAILURE
                        chunk14, max14 = None, self._offset + 1
                        if max14 <= self._input_size:
                            chunk14 = self._input[self._offset:max14]
                        if chunk14 is not None and Grammar.REGEX_28.search(chunk14):
                            address18 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                            self._offset = self._offset + 1
                        else:
                            address18 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::vis', '[\\d]'))
                        if address18 is not FAILURE:
                            elements5.append(address18)
                            address19 = FAILURE
                            chunk15, max15 = None, self._offset + 1
                            if max15 <= self._input_size:
                                chunk15 = self._input[self._offset:max15]
                            if chunk15 == '/':
                                address19 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                self._offset = self._offset + 1
                            else:
                                address19 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::vis', '"/"'))
                            if address19 is not FAILURE:
                                elements5.append(address19)
                                address20 = FAILURE
                                chunk16, max16 = None, self._offset + 1
                                if max16 <= self._input_size:
                                    chunk16 = self._input[self._offset:max16]
                                if chunk16 is not None and Grammar.REGEX_29.search(chunk16):
                                    address20 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                    self._offset = self._offset + 1
                                else:
                                    address20 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append(('METAR::vis', '[\\d]'))
                                if address20 is not FAILURE:
                                    elements5.append(address20)
                                    address21 = FAILURE
                                    chunk17, max17 = None, self._offset + 2
                                    if max17 <= self._input_size:
                                        chunk17 = self._input[self._offset:max17]
                                    if chunk17 == 'SM':
                                        address21 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                        self._offset = self._offset + 2
                                    else:
                                        address21 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append(('METAR::vis', '"SM"'))
                                    if address21 is not FAILURE:
                                        elements5.append(address21)
                                    else:
                                        elements5 = None
                                        self._offset = index13
                                else:
                                    elements5 = None
                                    self._offset = index13
                            else:
                                elements5 = None
                                self._offset = index13
                        else:
                            elements5 = None
                            self._offset = index13
                    else:
                        elements5 = None
                        self._offset = index13
                    if elements5 is None:
                        address2 = FAILURE
                    else:
                        address2 = TreeNode(self._input[index13:self._offset], index13, elements5)
                        self._offset = self._offset
                    if address2 is FAILURE:
                        self._offset = index3
                        chunk18, max18 = None, self._offset + 5
                        if max18 <= self._input_size:
                            chunk18 = self._input[self._offset:max18]
                        if chunk18 == 'CAVOK':
                            address2 = TreeNode(self._input[self._offset:self._offset + 5], self._offset, [])
                            self._offset = self._offset + 5
                        else:
                            address2 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::vis', '"CAVOK"'))
                        if address2 is FAILURE:
                            self._offset = index3
                            chunk19, max19 = None, self._offset + 4
                            if max19 <= self._input_size:
                                chunk19 = self._input[self._offset:max19]
                            if chunk19 == '////':
                                address2 = TreeNode(self._input[self._offset:self._offset + 4], self._offset, [])
                                self._offset = self._offset + 4
                            else:
                                address2 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::vis', '"////"'))
                            if address2 is FAILURE:
                                self._offset = index3
            if address2 is not FAILURE:
                elements0.append(address2)
                address22 = FAILURE
                index14 = self._offset
                address22 = self._read_varvis()
                if address22 is FAILURE:
                    address22 = TreeNode(self._input[index14:index14], index14, [])
                    self._offset = index14
                if address22 is not FAILURE:
                    elements0.append(address22)
                else:
                    elements0 = None
                    self._offset = index2
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode6(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['vis'][index0] = (address0, self._offset)
        return address0

    def _read_varvis(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['varvis'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0, max0 = None, self._offset + 1
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 is not None and Grammar.REGEX_30.search(chunk0):
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::varvis', '[\\d]'))
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk1, max1 = None, self._offset + 1
                if max1 <= self._input_size:
                    chunk1 = self._input[self._offset:max1]
                if chunk1 is not None and Grammar.REGEX_31.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::varvis', '[\\d]'))
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk2, max2 = None, self._offset + 1
                    if max2 <= self._input_size:
                        chunk2 = self._input[self._offset:max2]
                    if chunk2 is not None and Grammar.REGEX_32.search(chunk2):
                        address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::varvis', '[\\d]'))
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        chunk3, max3 = None, self._offset + 1
                        if max3 <= self._input_size:
                            chunk3 = self._input[self._offset:max3]
                        if chunk3 is not None and Grammar.REGEX_33.search(chunk3):
                            address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::varvis', '[\\d]'))
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            index2 = self._offset
                            chunk4, max4 = None, self._offset + 1
                            if max4 <= self._input_size:
                                chunk4 = self._input[self._offset:max4]
                            if chunk4 is not None and Grammar.REGEX_34.search(chunk4):
                                address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                self._offset = self._offset + 1
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::varvis', '[NSEW]'))
                            if address6 is FAILURE:
                                address6 = TreeNode(self._input[index2:index2], index2, [])
                                self._offset = index2
                            if address6 is not FAILURE:
                                elements0.append(address6)
                                address7 = FAILURE
                                index3 = self._offset
                                chunk5, max5 = None, self._offset + 1
                                if max5 <= self._input_size:
                                    chunk5 = self._input[self._offset:max5]
                                if chunk5 is not None and Grammar.REGEX_35.search(chunk5):
                                    address7 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                    self._offset = self._offset + 1
                                else:
                                    address7 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append(('METAR::varvis', '[NSEW]'))
                                if address7 is FAILURE:
                                    address7 = TreeNode(self._input[index3:index3], index3, [])
                                    self._offset = index3
                                if address7 is not FAILURE:
                                    elements0.append(address7)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode7(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['varvis'][index0] = (address0, self._offset)
        return address0

    def _read_run(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['run'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0, address1 = self._offset, [], None
        while True:
            index2, elements1 = self._offset, []
            address2 = FAILURE
            address2 = self._read_sep()
            if address2 is not FAILURE:
                elements1.append(address2)
                address3 = FAILURE
                chunk0, max0 = None, self._offset + 1
                if max0 <= self._input_size:
                    chunk0 = self._input[self._offset:max0]
                if chunk0 == 'R':
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::run', '"R"'))
                if address3 is not FAILURE:
                    elements1.append(address3)
                    address4 = FAILURE
                    index3 = self._offset
                    chunk1, max1 = None, self._offset + 1
                    if max1 <= self._input_size:
                        chunk1 = self._input[self._offset:max1]
                    if chunk1 is not None and Grammar.REGEX_36.search(chunk1):
                        address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::run', '[LRC]'))
                    if address4 is FAILURE:
                        address4 = TreeNode(self._input[index3:index3], index3, [])
                        self._offset = index3
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        address5 = FAILURE
                        chunk2, max2 = None, self._offset + 1
                        if max2 <= self._input_size:
                            chunk2 = self._input[self._offset:max2]
                        if chunk2 is not None and Grammar.REGEX_37.search(chunk2):
                            address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::run', '[\\d]'))
                        if address5 is not FAILURE:
                            elements1.append(address5)
                            address6 = FAILURE
                            chunk3, max3 = None, self._offset + 1
                            if max3 <= self._input_size:
                                chunk3 = self._input[self._offset:max3]
                            if chunk3 is not None and Grammar.REGEX_38.search(chunk3):
                                address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                self._offset = self._offset + 1
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::run', '[\\d]'))
                            if address6 is not FAILURE:
                                elements1.append(address6)
                                address7 = FAILURE
                                index4 = self._offset
                                chunk4, max4 = None, self._offset + 1
                                if max4 <= self._input_size:
                                    chunk4 = self._input[self._offset:max4]
                                if chunk4 is not None and Grammar.REGEX_39.search(chunk4):
                                    address7 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                    self._offset = self._offset + 1
                                else:
                                    address7 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append(('METAR::run', '[LRC]'))
                                if address7 is FAILURE:
                                    address7 = TreeNode(self._input[index4:index4], index4, [])
                                    self._offset = index4
                                if address7 is not FAILURE:
                                    elements1.append(address7)
                                    address8 = FAILURE
                                    chunk5, max5 = None, self._offset + 1
                                    if max5 <= self._input_size:
                                        chunk5 = self._input[self._offset:max5]
                                    if chunk5 == '/':
                                        address8 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                        self._offset = self._offset + 1
                                    else:
                                        address8 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append(('METAR::run', '"/"'))
                                    if address8 is not FAILURE:
                                        elements1.append(address8)
                                        address9 = FAILURE
                                        index5 = self._offset
                                        index6, elements2 = self._offset, []
                                        address10 = FAILURE
                                        chunk6, max6 = None, self._offset + 1
                                        if max6 <= self._input_size:
                                            chunk6 = self._input[self._offset:max6]
                                        if chunk6 is not None and Grammar.REGEX_40.search(chunk6):
                                            address10 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                            self._offset = self._offset + 1
                                        else:
                                            address10 = FAILURE
                                            if self._offset > self._failure:
                                                self._failure = self._offset
                                                self._expected = []
                                            if self._offset == self._failure:
                                                self._expected.append(('METAR::run', '[\\d]'))
                                        if address10 is not FAILURE:
                                            elements2.append(address10)
                                            address11 = FAILURE
                                            chunk7, max7 = None, self._offset + 1
                                            if max7 <= self._input_size:
                                                chunk7 = self._input[self._offset:max7]
                                            if chunk7 is not None and Grammar.REGEX_41.search(chunk7):
                                                address11 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                self._offset = self._offset + 1
                                            else:
                                                address11 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append(('METAR::run', '[\\d]'))
                                            if address11 is not FAILURE:
                                                elements2.append(address11)
                                                address12 = FAILURE
                                                chunk8, max8 = None, self._offset + 1
                                                if max8 <= self._input_size:
                                                    chunk8 = self._input[self._offset:max8]
                                                if chunk8 is not None and Grammar.REGEX_42.search(chunk8):
                                                    address12 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                    self._offset = self._offset + 1
                                                else:
                                                    address12 = FAILURE
                                                    if self._offset > self._failure:
                                                        self._failure = self._offset
                                                        self._expected = []
                                                    if self._offset == self._failure:
                                                        self._expected.append(('METAR::run', '[\\d]'))
                                                if address12 is not FAILURE:
                                                    elements2.append(address12)
                                                    address13 = FAILURE
                                                    chunk9, max9 = None, self._offset + 1
                                                    if max9 <= self._input_size:
                                                        chunk9 = self._input[self._offset:max9]
                                                    if chunk9 is not None and Grammar.REGEX_43.search(chunk9):
                                                        address13 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                        self._offset = self._offset + 1
                                                    else:
                                                        address13 = FAILURE
                                                        if self._offset > self._failure:
                                                            self._failure = self._offset
                                                            self._expected = []
                                                        if self._offset == self._failure:
                                                            self._expected.append(('METAR::run', '[\\d]'))
                                                    if address13 is not FAILURE:
                                                        elements2.append(address13)
                                                        address14 = FAILURE
                                                        chunk10, max10 = None, self._offset + 1
                                                        if max10 <= self._input_size:
                                                            chunk10 = self._input[self._offset:max10]
                                                        if chunk10 == 'V':
                                                            address14 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                            self._offset = self._offset + 1
                                                        else:
                                                            address14 = FAILURE
                                                            if self._offset > self._failure:
                                                                self._failure = self._offset
                                                                self._expected = []
                                                            if self._offset == self._failure:
                                                                self._expected.append(('METAR::run', '"V"'))
                                                        if address14 is not FAILURE:
                                                            elements2.append(address14)
                                                        else:
                                                            elements2 = None
                                                            self._offset = index6
                                                    else:
                                                        elements2 = None
                                                        self._offset = index6
                                                else:
                                                    elements2 = None
                                                    self._offset = index6
                                            else:
                                                elements2 = None
                                                self._offset = index6
                                        else:
                                            elements2 = None
                                            self._offset = index6
                                        if elements2 is None:
                                            address9 = FAILURE
                                        else:
                                            address9 = TreeNode(self._input[index6:self._offset], index6, elements2)
                                            self._offset = self._offset
                                        if address9 is FAILURE:
                                            address9 = TreeNode(self._input[index5:index5], index5, [])
                                            self._offset = index5
                                        if address9 is not FAILURE:
                                            elements1.append(address9)
                                            address15 = FAILURE
                                            index7 = self._offset
                                            chunk11, max11 = None, self._offset + 1
                                            if max11 <= self._input_size:
                                                chunk11 = self._input[self._offset:max11]
                                            if chunk11 is not None and Grammar.REGEX_44.search(chunk11):
                                                address15 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                self._offset = self._offset + 1
                                            else:
                                                address15 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append(('METAR::run', '["M" / "P"]'))
                                            if address15 is FAILURE:
                                                address15 = TreeNode(self._input[index7:index7], index7, [])
                                                self._offset = index7
                                            if address15 is not FAILURE:
                                                elements1.append(address15)
                                                address16 = FAILURE
                                                chunk12, max12 = None, self._offset + 1
                                                if max12 <= self._input_size:
                                                    chunk12 = self._input[self._offset:max12]
                                                if chunk12 is not None and Grammar.REGEX_45.search(chunk12):
                                                    address16 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                    self._offset = self._offset + 1
                                                else:
                                                    address16 = FAILURE
                                                    if self._offset > self._failure:
                                                        self._failure = self._offset
                                                        self._expected = []
                                                    if self._offset == self._failure:
                                                        self._expected.append(('METAR::run', '[\\d]'))
                                                if address16 is not FAILURE:
                                                    elements1.append(address16)
                                                    address17 = FAILURE
                                                    chunk13, max13 = None, self._offset + 1
                                                    if max13 <= self._input_size:
                                                        chunk13 = self._input[self._offset:max13]
                                                    if chunk13 is not None and Grammar.REGEX_46.search(chunk13):
                                                        address17 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                        self._offset = self._offset + 1
                                                    else:
                                                        address17 = FAILURE
                                                        if self._offset > self._failure:
                                                            self._failure = self._offset
                                                            self._expected = []
                                                        if self._offset == self._failure:
                                                            self._expected.append(('METAR::run', '[\\d]'))
                                                    if address17 is not FAILURE:
                                                        elements1.append(address17)
                                                        address18 = FAILURE
                                                        chunk14, max14 = None, self._offset + 1
                                                        if max14 <= self._input_size:
                                                            chunk14 = self._input[self._offset:max14]
                                                        if chunk14 is not None and Grammar.REGEX_47.search(chunk14):
                                                            address18 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                            self._offset = self._offset + 1
                                                        else:
                                                            address18 = FAILURE
                                                            if self._offset > self._failure:
                                                                self._failure = self._offset
                                                                self._expected = []
                                                            if self._offset == self._failure:
                                                                self._expected.append(('METAR::run', '[\\d]'))
                                                        if address18 is not FAILURE:
                                                            elements1.append(address18)
                                                            address19 = FAILURE
                                                            chunk15, max15 = None, self._offset + 1
                                                            if max15 <= self._input_size:
                                                                chunk15 = self._input[self._offset:max15]
                                                            if chunk15 is not None and Grammar.REGEX_48.search(chunk15):
                                                                address19 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                                self._offset = self._offset + 1
                                                            else:
                                                                address19 = FAILURE
                                                                if self._offset > self._failure:
                                                                    self._failure = self._offset
                                                                    self._expected = []
                                                                if self._offset == self._failure:
                                                                    self._expected.append(('METAR::run', '[\\d]'))
                                                            if address19 is not FAILURE:
                                                                elements1.append(address19)
                                                                address20 = FAILURE
                                                                index8 = self._offset
                                                                chunk16, max16 = None, self._offset + 2
                                                                if max16 <= self._input_size:
                                                                    chunk16 = self._input[self._offset:max16]
                                                                if chunk16 == 'FT':
                                                                    address20 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                    self._offset = self._offset + 2
                                                                else:
                                                                    address20 = FAILURE
                                                                    if self._offset > self._failure:
                                                                        self._failure = self._offset
                                                                        self._expected = []
                                                                    if self._offset == self._failure:
                                                                        self._expected.append(('METAR::run', '"FT"'))
                                                                if address20 is FAILURE:
                                                                    address20 = TreeNode(self._input[index8:index8], index8, [])
                                                                    self._offset = index8
                                                                if address20 is not FAILURE:
                                                                    elements1.append(address20)
                                                                    address21 = FAILURE
                                                                    index9 = self._offset
                                                                    index10, elements3 = self._offset, []
                                                                    address22 = FAILURE
                                                                    index11 = self._offset
                                                                    chunk17, max17 = None, self._offset + 1
                                                                    if max17 <= self._input_size:
                                                                        chunk17 = self._input[self._offset:max17]
                                                                    if chunk17 == '/':
                                                                        address22 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                                        self._offset = self._offset + 1
                                                                    else:
                                                                        address22 = FAILURE
                                                                        if self._offset > self._failure:
                                                                            self._failure = self._offset
                                                                            self._expected = []
                                                                        if self._offset == self._failure:
                                                                            self._expected.append(('METAR::run', '"/"'))
                                                                    if address22 is FAILURE:
                                                                        address22 = TreeNode(self._input[index11:index11], index11, [])
                                                                        self._offset = index11
                                                                    if address22 is not FAILURE:
                                                                        elements3.append(address22)
                                                                        address23 = FAILURE
                                                                        chunk18, max18 = None, self._offset + 1
                                                                        if max18 <= self._input_size:
                                                                            chunk18 = self._input[self._offset:max18]
                                                                        if chunk18 is not None and Grammar.REGEX_49.search(chunk18):
                                                                            address23 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                                                            self._offset = self._offset + 1
                                                                        else:
                                                                            address23 = FAILURE
                                                                            if self._offset > self._failure:
                                                                                self._failure = self._offset
                                                                                self._expected = []
                                                                            if self._offset == self._failure:
                                                                                self._expected.append(('METAR::run', '[UDN]'))
                                                                        if address23 is not FAILURE:
                                                                            elements3.append(address23)
                                                                        else:
                                                                            elements3 = None
                                                                            self._offset = index10
                                                                    else:
                                                                        elements3 = None
                                                                        self._offset = index10
                                                                    if elements3 is None:
                                                                        address21 = FAILURE
                                                                    else:
                                                                        address21 = TreeNode(self._input[index10:self._offset], index10, elements3)
                                                                        self._offset = self._offset
                                                                    if address21 is FAILURE:
                                                                        address21 = TreeNode(self._input[index9:index9], index9, [])
                                                                        self._offset = index9
                                                                    if address21 is not FAILURE:
                                                                        elements1.append(address21)
                                                                    else:
                                                                        elements1 = None
                                                                        self._offset = index2
                                                                else:
                                                                    elements1 = None
                                                                    self._offset = index2
                                                            else:
                                                                elements1 = None
                                                                self._offset = index2
                                                        else:
                                                            elements1 = None
                                                            self._offset = index2
                                                    else:
                                                        elements1 = None
                                                        self._offset = index2
                                                else:
                                                    elements1 = None
                                                    self._offset = index2
                                            else:
                                                elements1 = None
                                                self._offset = index2
                                        else:
                                            elements1 = None
                                            self._offset = index2
                                    else:
                                        elements1 = None
                                        self._offset = index2
                                else:
                                    elements1 = None
                                    self._offset = index2
                            else:
                                elements1 = None
                                self._offset = index2
                        else:
                            elements1 = None
                            self._offset = index2
                    else:
                        elements1 = None
                        self._offset = index2
                else:
                    elements1 = None
                    self._offset = index2
            else:
                elements1 = None
                self._offset = index2
            if elements1 is None:
                address1 = FAILURE
            else:
                address1 = TreeNode8(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            if address1 is not FAILURE:
                elements0.append(address1)
            else:
                break
        if len(elements0) >= 0:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        else:
            address0 = FAILURE
        self._cache['run'][index0] = (address0, self._offset)
        return address0

    def _read_curwx(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['curwx'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2 = self._offset
        index3, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0, max0 = None, self._offset + 2
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 == '//':
                address2 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                self._offset = self._offset + 2
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::curwx', '"//"'))
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index3
        else:
            elements0 = None
            self._offset = index3
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode9(self._input[index3:self._offset], index3, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index2
            index4, elements1 = self._offset, []
            address3 = FAILURE
            address3 = self._read_sep()
            if address3 is not FAILURE:
                elements1.append(address3)
                address4 = FAILURE
                chunk1, max1 = None, self._offset + 3
                if max1 <= self._input_size:
                    chunk1 = self._input[self._offset:max1]
                if chunk1 == 'NSW':
                    address4 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                    self._offset = self._offset + 3
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::curwx', '"NSW"'))
                if address4 is not FAILURE:
                    elements1.append(address4)
                else:
                    elements1 = None
                    self._offset = index4
            else:
                elements1 = None
                self._offset = index4
            if elements1 is None:
                address0 = FAILURE
            else:
                address0 = TreeNode10(self._input[index4:self._offset], index4, elements1)
                self._offset = self._offset
            if address0 is FAILURE:
                self._offset = index2
                index5, elements2, address5 = self._offset, [], None
                while True:
                    index6, elements3 = self._offset, []
                    address6 = FAILURE
                    address6 = self._read_sep()
                    if address6 is not FAILURE:
                        elements3.append(address6)
                        address7 = FAILURE
                        address7 = self._read_wx()
                        if address7 is not FAILURE:
                            elements3.append(address7)
                        else:
                            elements3 = None
                            self._offset = index6
                    else:
                        elements3 = None
                        self._offset = index6
                    if elements3 is None:
                        address5 = FAILURE
                    else:
                        address5 = TreeNode11(self._input[index6:self._offset], index6, elements3)
                        self._offset = self._offset
                    if address5 is not FAILURE:
                        elements2.append(address5)
                    else:
                        break
                if len(elements2) >= 0:
                    address0 = TreeNode(self._input[index5:self._offset], index5, elements2)
                    self._offset = self._offset
                else:
                    address0 = FAILURE
                if address0 is FAILURE:
                    self._offset = index2
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['curwx'][index0] = (address0, self._offset)
        return address0

    def _read_wx(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['wx'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        index3, elements1 = self._offset, []
        address2 = FAILURE
        chunk0, max0 = None, self._offset + 1
        if max0 <= self._input_size:
            chunk0 = self._input[self._offset:max0]
        if chunk0 is not None and Grammar.REGEX_50.search(chunk0):
            address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
            self._offset = self._offset + 1
        else:
            address2 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append(('METAR::wx', '[-+]'))
        if address2 is not FAILURE:
            elements1.append(address2)
            address3 = FAILURE
            index4 = self._offset
            chunk1, max1 = None, self._offset + 1
            if max1 <= self._input_size:
                chunk1 = self._input[self._offset:max1]
            if chunk1 == ' ':
                address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address3 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::wx', '" "'))
            if address3 is FAILURE:
                address3 = TreeNode(self._input[index4:index4], index4, [])
                self._offset = index4
            if address3 is not FAILURE:
                elements1.append(address3)
            else:
                elements1 = None
                self._offset = index3
        else:
            elements1 = None
            self._offset = index3
        if elements1 is None:
            address1 = FAILURE
        else:
            address1 = TreeNode(self._input[index3:self._offset], index3, elements1)
            self._offset = self._offset
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2, [])
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address4 = FAILURE
            index5 = self._offset
            chunk2, max2 = None, self._offset + 2
            if max2 <= self._input_size:
                chunk2 = self._input[self._offset:max2]
            if chunk2 == 'VC':
                address4 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                self._offset = self._offset + 2
            else:
                address4 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::wx', '"VC"'))
            if address4 is FAILURE:
                address4 = TreeNode(self._input[index5:index5], index5, [])
                self._offset = index5
            if address4 is not FAILURE:
                elements0.append(address4)
                address5 = FAILURE
                index6, elements2, address6 = self._offset, [], None
                while True:
                    index7 = self._offset
                    chunk3, max3 = None, self._offset + 2
                    if max3 <= self._input_size:
                        chunk3 = self._input[self._offset:max3]
                    if chunk3 == 'MI':
                        address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                        self._offset = self._offset + 2
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::wx', '"MI"'))
                    if address6 is FAILURE:
                        self._offset = index7
                        chunk4, max4 = None, self._offset + 2
                        if max4 <= self._input_size:
                            chunk4 = self._input[self._offset:max4]
                        if chunk4 == 'BC':
                            address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                            self._offset = self._offset + 2
                        else:
                            address6 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::wx', '"BC"'))
                        if address6 is FAILURE:
                            self._offset = index7
                            chunk5, max5 = None, self._offset + 2
                            if max5 <= self._input_size:
                                chunk5 = self._input[self._offset:max5]
                            if chunk5 == 'PR':
                                address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                self._offset = self._offset + 2
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::wx', '"PR"'))
                            if address6 is FAILURE:
                                self._offset = index7
                                chunk6, max6 = None, self._offset + 2
                                if max6 <= self._input_size:
                                    chunk6 = self._input[self._offset:max6]
                                if chunk6 == 'DR':
                                    address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                    self._offset = self._offset + 2
                                else:
                                    address6 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append(('METAR::wx', '"DR"'))
                                if address6 is FAILURE:
                                    self._offset = index7
                                    chunk7, max7 = None, self._offset + 2
                                    if max7 <= self._input_size:
                                        chunk7 = self._input[self._offset:max7]
                                    if chunk7 == 'BL':
                                        address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                        self._offset = self._offset + 2
                                    else:
                                        address6 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append(('METAR::wx', '"BL"'))
                                    if address6 is FAILURE:
                                        self._offset = index7
                                        chunk8, max8 = None, self._offset + 2
                                        if max8 <= self._input_size:
                                            chunk8 = self._input[self._offset:max8]
                                        if chunk8 == 'SH':
                                            address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                            self._offset = self._offset + 2
                                        else:
                                            address6 = FAILURE
                                            if self._offset > self._failure:
                                                self._failure = self._offset
                                                self._expected = []
                                            if self._offset == self._failure:
                                                self._expected.append(('METAR::wx', '"SH"'))
                                        if address6 is FAILURE:
                                            self._offset = index7
                                            chunk9, max9 = None, self._offset + 2
                                            if max9 <= self._input_size:
                                                chunk9 = self._input[self._offset:max9]
                                            if chunk9 == 'TS':
                                                address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                self._offset = self._offset + 2
                                            else:
                                                address6 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append(('METAR::wx', '"TS"'))
                                            if address6 is FAILURE:
                                                self._offset = index7
                                                chunk10, max10 = None, self._offset + 2
                                                if max10 <= self._input_size:
                                                    chunk10 = self._input[self._offset:max10]
                                                if chunk10 == 'FZ':
                                                    address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                    self._offset = self._offset + 2
                                                else:
                                                    address6 = FAILURE
                                                    if self._offset > self._failure:
                                                        self._failure = self._offset
                                                        self._expected = []
                                                    if self._offset == self._failure:
                                                        self._expected.append(('METAR::wx', '"FZ"'))
                                                if address6 is FAILURE:
                                                    self._offset = index7
                                                    chunk11, max11 = None, self._offset + 2
                                                    if max11 <= self._input_size:
                                                        chunk11 = self._input[self._offset:max11]
                                                    if chunk11 == 'DZ':
                                                        address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                        self._offset = self._offset + 2
                                                    else:
                                                        address6 = FAILURE
                                                        if self._offset > self._failure:
                                                            self._failure = self._offset
                                                            self._expected = []
                                                        if self._offset == self._failure:
                                                            self._expected.append(('METAR::wx', '"DZ"'))
                                                    if address6 is FAILURE:
                                                        self._offset = index7
                                                        chunk12, max12 = None, self._offset + 2
                                                        if max12 <= self._input_size:
                                                            chunk12 = self._input[self._offset:max12]
                                                        if chunk12 == 'RA':
                                                            address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                            self._offset = self._offset + 2
                                                        else:
                                                            address6 = FAILURE
                                                            if self._offset > self._failure:
                                                                self._failure = self._offset
                                                                self._expected = []
                                                            if self._offset == self._failure:
                                                                self._expected.append(('METAR::wx', '"RA"'))
                                                        if address6 is FAILURE:
                                                            self._offset = index7
                                                            chunk13, max13 = None, self._offset + 2
                                                            if max13 <= self._input_size:
                                                                chunk13 = self._input[self._offset:max13]
                                                            if chunk13 == 'SN':
                                                                address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                self._offset = self._offset + 2
                                                            else:
                                                                address6 = FAILURE
                                                                if self._offset > self._failure:
                                                                    self._failure = self._offset
                                                                    self._expected = []
                                                                if self._offset == self._failure:
                                                                    self._expected.append(('METAR::wx', '"SN"'))
                                                            if address6 is FAILURE:
                                                                self._offset = index7
                                                                chunk14, max14 = None, self._offset + 2
                                                                if max14 <= self._input_size:
                                                                    chunk14 = self._input[self._offset:max14]
                                                                if chunk14 == 'SG':
                                                                    address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                    self._offset = self._offset + 2
                                                                else:
                                                                    address6 = FAILURE
                                                                    if self._offset > self._failure:
                                                                        self._failure = self._offset
                                                                        self._expected = []
                                                                    if self._offset == self._failure:
                                                                        self._expected.append(('METAR::wx', '"SG"'))
                                                                if address6 is FAILURE:
                                                                    self._offset = index7
                                                                    chunk15, max15 = None, self._offset + 2
                                                                    if max15 <= self._input_size:
                                                                        chunk15 = self._input[self._offset:max15]
                                                                    if chunk15 == 'PL':
                                                                        address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                        self._offset = self._offset + 2
                                                                    else:
                                                                        address6 = FAILURE
                                                                        if self._offset > self._failure:
                                                                            self._failure = self._offset
                                                                            self._expected = []
                                                                        if self._offset == self._failure:
                                                                            self._expected.append(('METAR::wx', '"PL"'))
                                                                    if address6 is FAILURE:
                                                                        self._offset = index7
                                                                        chunk16, max16 = None, self._offset + 2
                                                                        if max16 <= self._input_size:
                                                                            chunk16 = self._input[self._offset:max16]
                                                                        if chunk16 == 'GR':
                                                                            address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                            self._offset = self._offset + 2
                                                                        else:
                                                                            address6 = FAILURE
                                                                            if self._offset > self._failure:
                                                                                self._failure = self._offset
                                                                                self._expected = []
                                                                            if self._offset == self._failure:
                                                                                self._expected.append(('METAR::wx', '"GR"'))
                                                                        if address6 is FAILURE:
                                                                            self._offset = index7
                                                                            chunk17, max17 = None, self._offset + 2
                                                                            if max17 <= self._input_size:
                                                                                chunk17 = self._input[self._offset:max17]
                                                                            if chunk17 == 'GS':
                                                                                address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                self._offset = self._offset + 2
                                                                            else:
                                                                                address6 = FAILURE
                                                                                if self._offset > self._failure:
                                                                                    self._failure = self._offset
                                                                                    self._expected = []
                                                                                if self._offset == self._failure:
                                                                                    self._expected.append(('METAR::wx', '"GS"'))
                                                                            if address6 is FAILURE:
                                                                                self._offset = index7
                                                                                chunk18, max18 = None, self._offset + 2
                                                                                if max18 <= self._input_size:
                                                                                    chunk18 = self._input[self._offset:max18]
                                                                                if chunk18 == 'UP':
                                                                                    address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                    self._offset = self._offset + 2
                                                                                else:
                                                                                    address6 = FAILURE
                                                                                    if self._offset > self._failure:
                                                                                        self._failure = self._offset
                                                                                        self._expected = []
                                                                                    if self._offset == self._failure:
                                                                                        self._expected.append(('METAR::wx', '"UP"'))
                                                                                if address6 is FAILURE:
                                                                                    self._offset = index7
                                                                                    chunk19, max19 = None, self._offset + 2
                                                                                    if max19 <= self._input_size:
                                                                                        chunk19 = self._input[self._offset:max19]
                                                                                    if chunk19 == 'BR':
                                                                                        address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                        self._offset = self._offset + 2
                                                                                    else:
                                                                                        address6 = FAILURE
                                                                                        if self._offset > self._failure:
                                                                                            self._failure = self._offset
                                                                                            self._expected = []
                                                                                        if self._offset == self._failure:
                                                                                            self._expected.append(('METAR::wx', '"BR"'))
                                                                                    if address6 is FAILURE:
                                                                                        self._offset = index7
                                                                                        chunk20, max20 = None, self._offset + 2
                                                                                        if max20 <= self._input_size:
                                                                                            chunk20 = self._input[self._offset:max20]
                                                                                        if chunk20 == 'FG':
                                                                                            address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                            self._offset = self._offset + 2
                                                                                        else:
                                                                                            address6 = FAILURE
                                                                                            if self._offset > self._failure:
                                                                                                self._failure = self._offset
                                                                                                self._expected = []
                                                                                            if self._offset == self._failure:
                                                                                                self._expected.append(('METAR::wx', '"FG"'))
                                                                                        if address6 is FAILURE:
                                                                                            self._offset = index7
                                                                                            chunk21, max21 = None, self._offset + 2
                                                                                            if max21 <= self._input_size:
                                                                                                chunk21 = self._input[self._offset:max21]
                                                                                            if chunk21 == 'FU':
                                                                                                address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                self._offset = self._offset + 2
                                                                                            else:
                                                                                                address6 = FAILURE
                                                                                                if self._offset > self._failure:
                                                                                                    self._failure = self._offset
                                                                                                    self._expected = []
                                                                                                if self._offset == self._failure:
                                                                                                    self._expected.append(('METAR::wx', '"FU"'))
                                                                                            if address6 is FAILURE:
                                                                                                self._offset = index7
                                                                                                chunk22, max22 = None, self._offset + 2
                                                                                                if max22 <= self._input_size:
                                                                                                    chunk22 = self._input[self._offset:max22]
                                                                                                if chunk22 == 'VA':
                                                                                                    address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                    self._offset = self._offset + 2
                                                                                                else:
                                                                                                    address6 = FAILURE
                                                                                                    if self._offset > self._failure:
                                                                                                        self._failure = self._offset
                                                                                                        self._expected = []
                                                                                                    if self._offset == self._failure:
                                                                                                        self._expected.append(('METAR::wx', '"VA"'))
                                                                                                if address6 is FAILURE:
                                                                                                    self._offset = index7
                                                                                                    chunk23, max23 = None, self._offset + 2
                                                                                                    if max23 <= self._input_size:
                                                                                                        chunk23 = self._input[self._offset:max23]
                                                                                                    if chunk23 == 'DU':
                                                                                                        address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                        self._offset = self._offset + 2
                                                                                                    else:
                                                                                                        address6 = FAILURE
                                                                                                        if self._offset > self._failure:
                                                                                                            self._failure = self._offset
                                                                                                            self._expected = []
                                                                                                        if self._offset == self._failure:
                                                                                                            self._expected.append(('METAR::wx', '"DU"'))
                                                                                                    if address6 is FAILURE:
                                                                                                        self._offset = index7
                                                                                                        chunk24, max24 = None, self._offset + 2
                                                                                                        if max24 <= self._input_size:
                                                                                                            chunk24 = self._input[self._offset:max24]
                                                                                                        if chunk24 == 'SA':
                                                                                                            address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                            self._offset = self._offset + 2
                                                                                                        else:
                                                                                                            address6 = FAILURE
                                                                                                            if self._offset > self._failure:
                                                                                                                self._failure = self._offset
                                                                                                                self._expected = []
                                                                                                            if self._offset == self._failure:
                                                                                                                self._expected.append(('METAR::wx', '"SA"'))
                                                                                                        if address6 is FAILURE:
                                                                                                            self._offset = index7
                                                                                                            chunk25, max25 = None, self._offset + 2
                                                                                                            if max25 <= self._input_size:
                                                                                                                chunk25 = self._input[self._offset:max25]
                                                                                                            if chunk25 == 'HZ':
                                                                                                                address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                                self._offset = self._offset + 2
                                                                                                            else:
                                                                                                                address6 = FAILURE
                                                                                                                if self._offset > self._failure:
                                                                                                                    self._failure = self._offset
                                                                                                                    self._expected = []
                                                                                                                if self._offset == self._failure:
                                                                                                                    self._expected.append(('METAR::wx', '"HZ"'))
                                                                                                            if address6 is FAILURE:
                                                                                                                self._offset = index7
                                                                                                                chunk26, max26 = None, self._offset + 2
                                                                                                                if max26 <= self._input_size:
                                                                                                                    chunk26 = self._input[self._offset:max26]
                                                                                                                if chunk26 == 'PO':
                                                                                                                    address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                                    self._offset = self._offset + 2
                                                                                                                else:
                                                                                                                    address6 = FAILURE
                                                                                                                    if self._offset > self._failure:
                                                                                                                        self._failure = self._offset
                                                                                                                        self._expected = []
                                                                                                                    if self._offset == self._failure:
                                                                                                                        self._expected.append(('METAR::wx', '"PO"'))
                                                                                                                if address6 is FAILURE:
                                                                                                                    self._offset = index7
                                                                                                                    chunk27, max27 = None, self._offset + 2
                                                                                                                    if max27 <= self._input_size:
                                                                                                                        chunk27 = self._input[self._offset:max27]
                                                                                                                    if chunk27 == 'SQ':
                                                                                                                        address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                                        self._offset = self._offset + 2
                                                                                                                    else:
                                                                                                                        address6 = FAILURE
                                                                                                                        if self._offset > self._failure:
                                                                                                                            self._failure = self._offset
                                                                                                                            self._expected = []
                                                                                                                        if self._offset == self._failure:
                                                                                                                            self._expected.append(('METAR::wx', '"SQ"'))
                                                                                                                    if address6 is FAILURE:
                                                                                                                        self._offset = index7
                                                                                                                        chunk28, max28 = None, self._offset + 2
                                                                                                                        if max28 <= self._input_size:
                                                                                                                            chunk28 = self._input[self._offset:max28]
                                                                                                                        if chunk28 == 'FC':
                                                                                                                            address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                                            self._offset = self._offset + 2
                                                                                                                        else:
                                                                                                                            address6 = FAILURE
                                                                                                                            if self._offset > self._failure:
                                                                                                                                self._failure = self._offset
                                                                                                                                self._expected = []
                                                                                                                            if self._offset == self._failure:
                                                                                                                                self._expected.append(('METAR::wx', '"FC"'))
                                                                                                                        if address6 is FAILURE:
                                                                                                                            self._offset = index7
                                                                                                                            chunk29, max29 = None, self._offset + 2
                                                                                                                            if max29 <= self._input_size:
                                                                                                                                chunk29 = self._input[self._offset:max29]
                                                                                                                            if chunk29 == 'SS':
                                                                                                                                address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                                                self._offset = self._offset + 2
                                                                                                                            else:
                                                                                                                                address6 = FAILURE
                                                                                                                                if self._offset > self._failure:
                                                                                                                                    self._failure = self._offset
                                                                                                                                    self._expected = []
                                                                                                                                if self._offset == self._failure:
                                                                                                                                    self._expected.append(('METAR::wx', '"SS"'))
                                                                                                                            if address6 is FAILURE:
                                                                                                                                self._offset = index7
                                                                                                                                chunk30, max30 = None, self._offset + 2
                                                                                                                                if max30 <= self._input_size:
                                                                                                                                    chunk30 = self._input[self._offset:max30]
                                                                                                                                if chunk30 == 'DS':
                                                                                                                                    address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                                                    self._offset = self._offset + 2
                                                                                                                                else:
                                                                                                                                    address6 = FAILURE
                                                                                                                                    if self._offset > self._failure:
                                                                                                                                        self._failure = self._offset
                                                                                                                                        self._expected = []
                                                                                                                                    if self._offset == self._failure:
                                                                                                                                        self._expected.append(('METAR::wx', '"DS"'))
                                                                                                                                if address6 is FAILURE:
                                                                                                                                    self._offset = index7
                                                                                                                                    chunk31, max31 = None, self._offset + 2
                                                                                                                                    if max31 <= self._input_size:
                                                                                                                                        chunk31 = self._input[self._offset:max31]
                                                                                                                                    if chunk31 == 'IC':
                                                                                                                                        address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                                                        self._offset = self._offset + 2
                                                                                                                                    else:
                                                                                                                                        address6 = FAILURE
                                                                                                                                        if self._offset > self._failure:
                                                                                                                                            self._failure = self._offset
                                                                                                                                            self._expected = []
                                                                                                                                        if self._offset == self._failure:
                                                                                                                                            self._expected.append(('METAR::wx', '"IC"'))
                                                                                                                                    if address6 is FAILURE:
                                                                                                                                        self._offset = index7
                                                                                                                                        chunk32, max32 = None, self._offset + 2
                                                                                                                                        if max32 <= self._input_size:
                                                                                                                                            chunk32 = self._input[self._offset:max32]
                                                                                                                                        if chunk32 == 'PY':
                                                                                                                                            address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                                                                                                                            self._offset = self._offset + 2
                                                                                                                                        else:
                                                                                                                                            address6 = FAILURE
                                                                                                                                            if self._offset > self._failure:
                                                                                                                                                self._failure = self._offset
                                                                                                                                                self._expected = []
                                                                                                                                            if self._offset == self._failure:
                                                                                                                                                self._expected.append(('METAR::wx', '"PY"'))
                                                                                                                                        if address6 is FAILURE:
                                                                                                                                            self._offset = index7
                    if address6 is not FAILURE:
                        elements2.append(address6)
                    else:
                        break
                if len(elements2) >= 1:
                    address5 = TreeNode(self._input[index6:self._offset], index6, elements2)
                    self._offset = self._offset
                else:
                    address5 = FAILURE
                if address5 is not FAILURE:
                    elements0.append(address5)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['wx'][index0] = (address0, self._offset)
        return address0

    def _read_skyc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['skyc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0, address1 = self._offset, [], None
        while True:
            index3, elements1 = self._offset, []
            address2 = FAILURE
            address2 = self._read_sep()
            if address2 is not FAILURE:
                elements1.append(address2)
                address3 = FAILURE
                address3 = self._read_cover()
                if address3 is not FAILURE:
                    elements1.append(address3)
                else:
                    elements1 = None
                    self._offset = index3
            else:
                elements1 = None
                self._offset = index3
            if elements1 is None:
                address1 = FAILURE
            else:
                address1 = TreeNode12(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            if address1 is not FAILURE:
                elements0.append(address1)
            else:
                break
        if len(elements0) >= 0:
            address0 = TreeNode(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        else:
            address0 = FAILURE
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['skyc'][index0] = (address0, self._offset)
        return address0

    def _read_cover(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['cover'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        chunk0, max0 = None, self._offset + 3
        if max0 <= self._input_size:
            chunk0 = self._input[self._offset:max0]
        if chunk0 == 'FEW':
            address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
            self._offset = self._offset + 3
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append(('METAR::cover', '"FEW"'))
        if address1 is FAILURE:
            self._offset = index3
            chunk1, max1 = None, self._offset + 3
            if max1 <= self._input_size:
                chunk1 = self._input[self._offset:max1]
            if chunk1 == 'SCT':
                address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                self._offset = self._offset + 3
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::cover', '"SCT"'))
            if address1 is FAILURE:
                self._offset = index3
                chunk2, max2 = None, self._offset + 3
                if max2 <= self._input_size:
                    chunk2 = self._input[self._offset:max2]
                if chunk2 == 'BKN':
                    address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                    self._offset = self._offset + 3
                else:
                    address1 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::cover', '"BKN"'))
                if address1 is FAILURE:
                    self._offset = index3
                    chunk3, max3 = None, self._offset + 3
                    if max3 <= self._input_size:
                        chunk3 = self._input[self._offset:max3]
                    if chunk3 == 'OVC':
                        address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                        self._offset = self._offset + 3
                    else:
                        address1 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::cover', '"OVC"'))
                    if address1 is FAILURE:
                        self._offset = index3
                        chunk4, max4 = None, self._offset + 2
                        if max4 <= self._input_size:
                            chunk4 = self._input[self._offset:max4]
                        if chunk4 == 'VV':
                            address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                            self._offset = self._offset + 2
                        else:
                            address1 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::cover', '"VV"'))
                        if address1 is FAILURE:
                            self._offset = index3
                            chunk5, max5 = None, self._offset + 3
                            if max5 <= self._input_size:
                                chunk5 = self._input[self._offset:max5]
                            if chunk5 == '///':
                                address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                                self._offset = self._offset + 3
                            else:
                                address1 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::cover', '"///"'))
                            if address1 is FAILURE:
                                self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index4 = self._offset
            index5, elements1, address3 = self._offset, [], None
            while True:
                chunk6, max6 = None, self._offset + 1
                if max6 <= self._input_size:
                    chunk6 = self._input[self._offset:max6]
                if chunk6 is not None and Grammar.REGEX_51.search(chunk6):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::cover', '[\\d]'))
                if address3 is not FAILURE:
                    elements1.append(address3)
                else:
                    break
            if len(elements1) >= 0:
                address2 = TreeNode(self._input[index5:self._offset], index5, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index4:index4], index4, [])
                self._offset = index4
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                index6 = self._offset
                index7 = self._offset
                chunk7, max7 = None, self._offset + 3
                if max7 <= self._input_size:
                    chunk7 = self._input[self._offset:max7]
                if chunk7 == 'TCU':
                    address4 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                    self._offset = self._offset + 3
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::cover', '"TCU"'))
                if address4 is FAILURE:
                    self._offset = index7
                    chunk8, max8 = None, self._offset + 2
                    if max8 <= self._input_size:
                        chunk8 = self._input[self._offset:max8]
                    if chunk8 == 'CB':
                        address4 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                        self._offset = self._offset + 2
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::cover', '"CB"'))
                    if address4 is FAILURE:
                        self._offset = index7
                        index8, elements2 = self._offset, []
                        address5 = FAILURE
                        chunk9, max9 = None, self._offset + 2
                        if max9 <= self._input_size:
                            chunk9 = self._input[self._offset:max9]
                        if chunk9 == '//':
                            address5 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                            self._offset = self._offset + 2
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::cover', '"//"'))
                        if address5 is not FAILURE:
                            elements2.append(address5)
                            address6 = FAILURE
                            index9 = self._offset
                            chunk10, max10 = None, self._offset + 1
                            if max10 <= self._input_size:
                                chunk10 = self._input[self._offset:max10]
                            if chunk10 == '/':
                                address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                self._offset = self._offset + 1
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::cover', '"/"'))
                            if address6 is FAILURE:
                                address6 = TreeNode(self._input[index9:index9], index9, [])
                                self._offset = index9
                            if address6 is not FAILURE:
                                elements2.append(address6)
                            else:
                                elements2 = None
                                self._offset = index8
                        else:
                            elements2 = None
                            self._offset = index8
                        if elements2 is None:
                            address4 = FAILURE
                        else:
                            address4 = TreeNode(self._input[index8:self._offset], index8, elements2)
                            self._offset = self._offset
                        if address4 is FAILURE:
                            self._offset = index7
                if address4 is FAILURE:
                    address4 = TreeNode(self._input[index6:index6], index6, [])
                    self._offset = index6
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index2
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index1
            index10 = self._offset
            chunk11, max11 = None, self._offset + 3
            if max11 <= self._input_size:
                chunk11 = self._input[self._offset:max11]
            if chunk11 == 'CLR':
                address0 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                self._offset = self._offset + 3
            else:
                address0 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::cover', '"CLR"'))
            if address0 is FAILURE:
                self._offset = index10
                chunk12, max12 = None, self._offset + 3
                if max12 <= self._input_size:
                    chunk12 = self._input[self._offset:max12]
                if chunk12 == 'SKC':
                    address0 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                    self._offset = self._offset + 3
                else:
                    address0 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::cover', '"SKC"'))
                if address0 is FAILURE:
                    self._offset = index10
                    chunk13, max13 = None, self._offset + 3
                    if max13 <= self._input_size:
                        chunk13 = self._input[self._offset:max13]
                    if chunk13 == 'NSC':
                        address0 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                        self._offset = self._offset + 3
                    else:
                        address0 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::cover', '"NSC"'))
                    if address0 is FAILURE:
                        self._offset = index10
                        chunk14, max14 = None, self._offset + 3
                        if max14 <= self._input_size:
                            chunk14 = self._input[self._offset:max14]
                        if chunk14 == 'NCD':
                            address0 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                            self._offset = self._offset + 3
                        else:
                            address0 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::cover', '"NCD"'))
                        if address0 is FAILURE:
                            self._offset = index10
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_wx()
                if address0 is FAILURE:
                    self._offset = index1
                    chunk15, max15 = None, self._offset + 2
                    if max15 <= self._input_size:
                        chunk15 = self._input[self._offset:max15]
                    if chunk15 == '//':
                        address0 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                        self._offset = self._offset + 2
                    else:
                        address0 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::cover', '"//"'))
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache['cover'][index0] = (address0, self._offset)
        return address0

    def _read_temp_dewp(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['temp_dewp'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            chunk0, max0 = None, self._offset + 2
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 == '//':
                address2 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                self._offset = self._offset + 2
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::temp_dewp', '"//"'))
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index3:index3], index3, [])
                self._offset = index3
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                address3 = self._read_temp()
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk1, max1 = None, self._offset + 1
                    if max1 <= self._input_size:
                        chunk1 = self._input[self._offset:max1]
                    if chunk1 == '/':
                        address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::temp_dewp', '"/"'))
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        address5 = self._read_dewp()
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            index4 = self._offset
                            chunk2, max2 = None, self._offset + 2
                            if max2 <= self._input_size:
                                chunk2 = self._input[self._offset:max2]
                            if chunk2 == '//':
                                address6 = TreeNode(self._input[self._offset:self._offset + 2], self._offset, [])
                                self._offset = self._offset + 2
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::temp_dewp', '"//"'))
                            if address6 is FAILURE:
                                address6 = TreeNode(self._input[index4:index4], index4, [])
                                self._offset = index4
                            if address6 is not FAILURE:
                                elements0.append(address6)
                            else:
                                elements0 = None
                                self._offset = index2
                        else:
                            elements0 = None
                            self._offset = index2
                    else:
                        elements0 = None
                        self._offset = index2
                else:
                    elements0 = None
                    self._offset = index2
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode13(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['temp_dewp'][index0] = (address0, self._offset)
        return address0

    def _read_temp(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['temp'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0, max0 = None, self._offset + 1
        if max0 <= self._input_size:
            chunk0 = self._input[self._offset:max0]
        if chunk0 is not None and Grammar.REGEX_52.search(chunk0):
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append(('METAR::temp', '[M]'))
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2, [])
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            chunk1, max1 = None, self._offset + 1
            if max1 <= self._input_size:
                chunk1 = self._input[self._offset:max1]
            if chunk1 is not None and Grammar.REGEX_53.search(chunk1):
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::temp', '[\\d]'))
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index3:index3], index3, [])
                self._offset = index3
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index4 = self._offset
                chunk2, max2 = None, self._offset + 1
                if max2 <= self._input_size:
                    chunk2 = self._input[self._offset:max2]
                if chunk2 is not None and Grammar.REGEX_54.search(chunk2):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::temp', '[\\d]'))
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index4:index4], index4, [])
                    self._offset = index4
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['temp'][index0] = (address0, self._offset)
        return address0

    def _read_dewp(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['dewp'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0, max0 = None, self._offset + 1
        if max0 <= self._input_size:
            chunk0 = self._input[self._offset:max0]
        if chunk0 is not None and Grammar.REGEX_55.search(chunk0):
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append(('METAR::dewp', '[M]'))
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2, [])
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            chunk1, max1 = None, self._offset + 1
            if max1 <= self._input_size:
                chunk1 = self._input[self._offset:max1]
            if chunk1 is not None and Grammar.REGEX_56.search(chunk1):
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::dewp', '[\\d]'))
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index3:index3], index3, [])
                self._offset = index3
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index4 = self._offset
                chunk2, max2 = None, self._offset + 1
                if max2 <= self._input_size:
                    chunk2 = self._input[self._offset:max2]
                if chunk2 is not None and Grammar.REGEX_57.search(chunk2):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::dewp', '[\\d]'))
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index4:index4], index4, [])
                    self._offset = index4
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['dewp'][index0] = (address0, self._offset)
        return address0

    def _read_altim(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['altim'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index3:index3], index3, [])
            self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0, max0 = None, self._offset + 1
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 is not None and Grammar.REGEX_58.search(chunk0):
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::altim', '["Q" / "A"]'))
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk1, max1 = None, self._offset + 1
                if max1 <= self._input_size:
                    chunk1 = self._input[self._offset:max1]
                if chunk1 is not None and Grammar.REGEX_59.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append(('METAR::altim', '[\\d]'))
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk2, max2 = None, self._offset + 1
                    if max2 <= self._input_size:
                        chunk2 = self._input[self._offset:max2]
                    if chunk2 is not None and Grammar.REGEX_60.search(chunk2):
                        address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::altim', '[\\d]'))
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        chunk3, max3 = None, self._offset + 1
                        if max3 <= self._input_size:
                            chunk3 = self._input[self._offset:max3]
                        if chunk3 is not None and Grammar.REGEX_61.search(chunk3):
                            address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append(('METAR::altim', '[\\d]'))
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            chunk4, max4 = None, self._offset + 1
                            if max4 <= self._input_size:
                                chunk4 = self._input[self._offset:max4]
                            if chunk4 is not None and Grammar.REGEX_62.search(chunk4):
                                address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                self._offset = self._offset + 1
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append(('METAR::altim', '[\\d]'))
                            if address6 is not FAILURE:
                                elements0.append(address6)
                                address7 = FAILURE
                                index4 = self._offset
                                chunk5, max5 = None, self._offset + 1
                                if max5 <= self._input_size:
                                    chunk5 = self._input[self._offset:max5]
                                if chunk5 == '=':
                                    address7 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                                    self._offset = self._offset + 1
                                else:
                                    address7 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append(('METAR::altim', '"="'))
                                if address7 is FAILURE:
                                    address7 = TreeNode(self._input[index4:index4], index4, [])
                                    self._offset = index4
                                if address7 is not FAILURE:
                                    elements0.append(address7)
                                else:
                                    elements0 = None
                                    self._offset = index2
                            else:
                                elements0 = None
                                self._offset = index2
                        else:
                            elements0 = None
                            self._offset = index2
                    else:
                        elements0 = None
                        self._offset = index2
                else:
                    elements0 = None
                    self._offset = index2
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['altim'][index0] = (address0, self._offset)
        return address0

    def _read_remarks(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['remarks'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index3:index3], index3, [])
            self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index4 = self._offset
            chunk0, max0 = None, self._offset + 3
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 == 'RMK':
                address2 = TreeNode(self._input[self._offset:self._offset + 3], self._offset, [])
                self._offset = self._offset + 3
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::remarks', '"RMK"'))
            if address2 is FAILURE:
                self._offset = index4
                index5, elements1, address3 = self._offset, [], None
                while True:
                    chunk1, max1 = None, self._offset + 5
                    if max1 <= self._input_size:
                        chunk1 = self._input[self._offset:max1]
                    if chunk1 == 'NOSIG':
                        address3 = TreeNode(self._input[self._offset:self._offset + 5], self._offset, [])
                        self._offset = self._offset + 5
                    else:
                        address3 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::remarks', '"NOSIG"'))
                    if address3 is not FAILURE:
                        elements1.append(address3)
                    else:
                        break
                if len(elements1) >= 0:
                    address2 = TreeNode(self._input[index5:self._offset], index5, elements1)
                    self._offset = self._offset
                else:
                    address2 = FAILURE
                if address2 is FAILURE:
                    self._offset = index4
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                index6, elements2, address5 = self._offset, [], None
                while True:
                    if self._offset < self._input_size:
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append(('METAR::remarks', '<any char>'))
                    if address5 is not FAILURE:
                        elements2.append(address5)
                    else:
                        break
                if len(elements2) >= 0:
                    address4 = TreeNode(self._input[index6:self._offset], index6, elements2)
                    self._offset = self._offset
                else:
                    address4 = FAILURE
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index2
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['remarks'][index0] = (address0, self._offset)
        return address0

    def _read_end(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['end'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index3:index3], index3, [])
            self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0, max0 = None, self._offset + 1
            if max0 <= self._input_size:
                chunk0 = self._input[self._offset:max0]
            if chunk0 == '=':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset, [])
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append(('METAR::end', '"="'))
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1, [])
            self._offset = index1
        self._cache['end'][index0] = (address0, self._offset)
        return address0


class Parser(Grammar):
    def __init__(self, input, actions, types):
        self._input = input
        self._input_size = len(input)
        self._actions = actions
        self._types = types
        self._offset = 0
        self._cache = defaultdict(dict)
        self._failure = 0
        self._expected = []

    def parse(self):
        tree = self._read_ob()
        if tree is not FAILURE and self._offset == self._input_size:
            return tree
        if not self._expected:
            self._failure = self._offset
            self._expected.append(('METAR', '<EOF>'))
        raise ParseError(format_error(self._input, self._failure, self._expected))


class ParseError(SyntaxError):
    pass


def parse(input, actions=None, types=None):
    parser = Parser(input, actions, types)
    return parser.parse()

def format_error(input, offset, expected):
    lines = input.split('\n')
    line_no, position = 0, 0

    while position <= offset:
        position += len(lines[line_no]) + 1
        line_no += 1

    line = lines[line_no - 1]
    message = 'Line ' + str(line_no) + ': expected one of:\n\n'

    for pair in expected:
        message += '    - ' + pair[1] + ' from ' + pair[0] + '\n'

    number = str(line_no)
    while len(number) < 6:
        number = ' ' + number

    message += '\n' + number + ' | ' + line + '\n'
    message += ' ' * (len(line) + 10 + offset - position)
    return message + '^'
