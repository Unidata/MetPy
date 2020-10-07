from collections import defaultdict
import re


class TreeNode:
    def __init__(self, text, offset, elements=None):
        self.text = text
        self.offset = offset
        self.elements = elements or []

    def __iter__(self):
        yield from self.elements


class TreeNode1(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
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
        super().__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode3(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode4(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
        self.wind_dir = elements[1]
        self.wind_spd = elements[2]


class TreeNode5(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode6(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode7(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode8(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode9(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
        self.sep = elements[0]


class TreeNode10(TreeNode):
    def __init__(self, text, offset, elements):
        super().__init__(text, offset, elements)
        self.sep = elements[0]
        self.temp = elements[2]
        self.dewp = elements[4]


class ParseError(SyntaxError):
    pass


FAILURE = object()


class Grammar:
    REGEX_1 = re.compile("^[0-9A-Z]")
    REGEX_2 = re.compile("^[0-9A-Z]")
    REGEX_3 = re.compile("^[0-9A-Z]")
    REGEX_4 = re.compile("^[0-9A-Z]")
    REGEX_5 = re.compile("^[\\d]")
    REGEX_6 = re.compile("^[\\d]")
    REGEX_7 = re.compile("^[\\d]")
    REGEX_8 = re.compile("^[\\d]")
    REGEX_9 = re.compile("^[\\d]")
    REGEX_10 = re.compile("^[\\d]")
    REGEX_11 = re.compile("^[\\d]")
    REGEX_12 = re.compile("^[\\d]")
    REGEX_13 = re.compile("^[\\d]")
    REGEX_14 = re.compile("^[\\d]")
    REGEX_15 = re.compile("^[\\d]")
    REGEX_16 = re.compile("^[\\d]")
    REGEX_17 = re.compile("^[\\d]")
    REGEX_18 = re.compile("^[\\d]")
    REGEX_19 = re.compile("^[\\d]")
    REGEX_20 = re.compile("^[\\d]")
    REGEX_21 = re.compile("^[\\d]")
    REGEX_22 = re.compile("^[\\d]")
    REGEX_23 = re.compile("^[\\d]")
    REGEX_24 = re.compile("^[\\d]")
    REGEX_25 = re.compile("^[\\d]")
    REGEX_26 = re.compile("^[\\d]")
    REGEX_27 = re.compile("^[LRC]")
    REGEX_28 = re.compile("^[\\d]")
    REGEX_29 = re.compile("^[\\d]")
    REGEX_30 = re.compile("^[LRC]")
    REGEX_31 = re.compile("^[\\d]")
    REGEX_32 = re.compile("^[\\d]")
    REGEX_33 = re.compile("^[\\d]")
    REGEX_34 = re.compile("^[\\d]")
    REGEX_35 = re.compile('^["M" \\/ "P"]')
    REGEX_36 = re.compile("^[\\d]")
    REGEX_37 = re.compile("^[\\d]")
    REGEX_38 = re.compile("^[\\d]")
    REGEX_39 = re.compile("^[\\d]")
    REGEX_40 = re.compile("^[-+]")
    REGEX_41 = re.compile("^[-+]")
    REGEX_42 = re.compile("^[\\d]")
    REGEX_43 = re.compile("^[M]")
    REGEX_44 = re.compile("^[\\d]")
    REGEX_45 = re.compile("^[\\d]")
    REGEX_46 = re.compile("^[M]")
    REGEX_47 = re.compile("^[\\d]")
    REGEX_48 = re.compile("^[\\d]")
    REGEX_49 = re.compile('^["Q" \\/ "A"]')
    REGEX_50 = re.compile("^[\\d]")
    REGEX_51 = re.compile("^[\\d]")
    REGEX_52 = re.compile("^[\\d]")
    REGEX_53 = re.compile("^[\\d]")

    def _read_ob(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["ob"].get(index0)
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
            address0 = TreeNode1(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["ob"][index0] = (address0, self._offset)
        return address0

    def _read_metar(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["metar"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        index3 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset : self._offset + 5]
        if chunk0 == "METAR":
            address1 = TreeNode(self._input[self._offset : self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"METAR"')
        if address1 is FAILURE:
            self._offset = index3
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset : self._offset + 5]
            if chunk1 == "SPECI":
                address1 = TreeNode(self._input[self._offset : self._offset + 5], self._offset)
                self._offset = self._offset + 5
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"SPECI"')
            if address1 is FAILURE:
                self._offset = index3
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2)
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index4 = self._offset
            index5 = self._offset
            chunk2 = None
            if self._offset < self._input_size:
                chunk2 = self._input[self._offset : self._offset + 5]
            if chunk2 == " AUTO":
                address2 = TreeNode(self._input[self._offset : self._offset + 5], self._offset)
                self._offset = self._offset + 5
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('" AUTO"')
            if address2 is FAILURE:
                self._offset = index5
                chunk3 = None
                if self._offset < self._input_size:
                    chunk3 = self._input[self._offset : self._offset + 4]
                if chunk3 == " COR":
                    address2 = TreeNode(
                        self._input[self._offset : self._offset + 4], self._offset
                    )
                    self._offset = self._offset + 4
                else:
                    address2 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('" COR"')
                if address2 is FAILURE:
                    self._offset = index5
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index4:index4], index4)
                self._offset = index4
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["metar"][index0] = (address0, self._offset)
        return address0

    def _read_sep(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["sep"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        remaining0, index1, elements0, address1 = 1, self._offset, [], True
        while address1 is not FAILURE:
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 1]
            if chunk0 == " ":
                address1 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('" "')
            if address1 is not FAILURE:
                elements0.append(address1)
                remaining0 -= 1
        if remaining0 <= 0:
            address0 = TreeNode(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        else:
            address0 = FAILURE
        self._cache["sep"][index0] = (address0, self._offset)
        return address0

    def _read_siteid(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["siteid"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2)
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 1]
            if chunk0 is not None and Grammar.REGEX_1.search(chunk0):
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append("[0-9A-Z]")
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset : self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_2.search(chunk1):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[0-9A-Z]")
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk2 = None
                    if self._offset < self._input_size:
                        chunk2 = self._input[self._offset : self._offset + 1]
                    if chunk2 is not None and Grammar.REGEX_3.search(chunk2):
                        address4 = TreeNode(
                            self._input[self._offset : self._offset + 1], self._offset
                        )
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append("[0-9A-Z]")
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        chunk3 = None
                        if self._offset < self._input_size:
                            chunk3 = self._input[self._offset : self._offset + 1]
                        if chunk3 is not None and Grammar.REGEX_4.search(chunk3):
                            address5 = TreeNode(
                                self._input[self._offset : self._offset + 1], self._offset
                            )
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append("[0-9A-Z]")
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
            address0 = TreeNode(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["siteid"][index0] = (address0, self._offset)
        return address0

    def _read_datetime(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["datetime"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 1, self._offset, [], True
            while address3 is not FAILURE:
                chunk0 = None
                if self._offset < self._input_size:
                    chunk0 = self._input[self._offset : self._offset + 1]
                if chunk0 is not None and Grammar.REGEX_5.search(chunk0):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2 : self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset : self._offset + 1]
                if chunk1 == "Z":
                    address4 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"Z"')
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
            address0 = TreeNode2(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["datetime"][index0] = (address0, self._offset)
        return address0

    def _read_auto(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["auto"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            index3 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 4]
            if chunk0 == "AUTO":
                address2 = TreeNode(self._input[self._offset : self._offset + 4], self._offset)
                self._offset = self._offset + 4
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"AUTO"')
            if address2 is FAILURE:
                self._offset = index3
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset : self._offset + 3]
                if chunk1 == "COR":
                    address2 = TreeNode(
                        self._input[self._offset : self._offset + 3], self._offset
                    )
                    self._offset = self._offset + 3
                else:
                    address2 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"COR"')
                if address2 is FAILURE:
                    self._offset = index3
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index2:index2], index2)
                self._offset = index2
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode3(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["auto"][index0] = (address0, self._offset)
        return address0

    def _read_wind(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["wind"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index3:index3], index3)
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
                    index4 = self._offset
                    address4 = self._read_gust()
                    if address4 is FAILURE:
                        address4 = TreeNode(self._input[index4:index4], index4)
                        self._offset = index4
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        index5 = self._offset
                        chunk0 = None
                        if self._offset < self._input_size:
                            chunk0 = self._input[self._offset : self._offset + 2]
                        if chunk0 == "KT":
                            address5 = TreeNode(
                                self._input[self._offset : self._offset + 2], self._offset
                            )
                            self._offset = self._offset + 2
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('"KT"')
                        if address5 is FAILURE:
                            self._offset = index5
                            chunk1 = None
                            if self._offset < self._input_size:
                                chunk1 = self._input[self._offset : self._offset + 3]
                            if chunk1 == "MPS":
                                address5 = TreeNode(
                                    self._input[self._offset : self._offset + 3], self._offset
                                )
                                self._offset = self._offset + 3
                            else:
                                address5 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('"MPS"')
                            if address5 is FAILURE:
                                self._offset = index5
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            index6 = self._offset
                            address6 = self._read_varwind()
                            if address6 is FAILURE:
                                address6 = TreeNode(self._input[index6:index6], index6)
                                self._offset = index6
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
            address0 = TreeNode4(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["wind"][index0] = (address0, self._offset)
        return address0

    def _read_wind_dir(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["wind_dir"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2 = self._offset
        index3, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset : self._offset + 1]
        if chunk0 is not None and Grammar.REGEX_6.search(chunk0):
            address1 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append("[\\d]")
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset : self._offset + 1]
            if chunk1 is not None and Grammar.REGEX_7.search(chunk1):
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append("[\\d]")
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset : self._offset + 1]
                if chunk2 is not None and Grammar.REGEX_8.search(chunk2):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
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
            address0 = TreeNode(self._input[index3 : self._offset], index3, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index2
            chunk3 = None
            if self._offset < self._input_size:
                chunk3 = self._input[self._offset : self._offset + 3]
            if chunk3 == "VAR":
                address0 = TreeNode(self._input[self._offset : self._offset + 3], self._offset)
                self._offset = self._offset + 3
            else:
                address0 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append("'VAR'")
            if address0 is FAILURE:
                self._offset = index2
                chunk4 = None
                if self._offset < self._input_size:
                    chunk4 = self._input[self._offset : self._offset + 3]
                if chunk4 == "VRB":
                    address0 = TreeNode(
                        self._input[self._offset : self._offset + 3], self._offset
                    )
                    self._offset = self._offset + 3
                else:
                    address0 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("'VRB'")
                if address0 is FAILURE:
                    self._offset = index2
                    chunk5 = None
                    if self._offset < self._input_size:
                        chunk5 = self._input[self._offset : self._offset + 3]
                    if chunk5 == "///":
                        address0 = TreeNode(
                            self._input[self._offset : self._offset + 3], self._offset
                        )
                        self._offset = self._offset + 3
                    else:
                        address0 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"///"')
                    if address0 is FAILURE:
                        self._offset = index2
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["wind_dir"][index0] = (address0, self._offset)
        return address0

    def _read_wind_spd(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["wind_spd"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2 = self._offset
        index3, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset : self._offset + 1]
        if chunk0 is not None and Grammar.REGEX_9.search(chunk0):
            address1 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append("[\\d]")
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset : self._offset + 1]
            if chunk1 is not None and Grammar.REGEX_10.search(chunk1):
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append("[\\d]")
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index4 = self._offset
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset : self._offset + 1]
                if chunk2 is not None and Grammar.REGEX_11.search(chunk2):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index4:index4], index4)
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
            address0 = TreeNode(self._input[index3 : self._offset], index3, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index2
            chunk3 = None
            if self._offset < self._input_size:
                chunk3 = self._input[self._offset : self._offset + 2]
            if chunk3 == "//":
                address0 = TreeNode(self._input[self._offset : self._offset + 2], self._offset)
                self._offset = self._offset + 2
            else:
                address0 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"//"')
            if address0 is FAILURE:
                self._offset = index2
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["wind_spd"][index0] = (address0, self._offset)
        return address0

    def _read_gust(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["gust"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset : self._offset + 1]
        if chunk0 == "G":
            address1 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"G"')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 1, self._offset, [], True
            while address3 is not FAILURE:
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset : self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_12.search(chunk1):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2 : self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["gust"][index0] = (address0, self._offset)
        return address0

    def _read_varwind(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["varwind"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_sep()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 1]
            if chunk0 is not None and Grammar.REGEX_13.search(chunk0):
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append("[\\d]")
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset : self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_14.search(chunk1):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk2 = None
                    if self._offset < self._input_size:
                        chunk2 = self._input[self._offset : self._offset + 1]
                    if chunk2 is not None and Grammar.REGEX_15.search(chunk2):
                        address4 = TreeNode(
                            self._input[self._offset : self._offset + 1], self._offset
                        )
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append("[\\d]")
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        chunk3 = None
                        if self._offset < self._input_size:
                            chunk3 = self._input[self._offset : self._offset + 1]
                        if chunk3 == "V":
                            address5 = TreeNode(
                                self._input[self._offset : self._offset + 1], self._offset
                            )
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('"V"')
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            chunk4 = None
                            if self._offset < self._input_size:
                                chunk4 = self._input[self._offset : self._offset + 1]
                            if chunk4 is not None and Grammar.REGEX_16.search(chunk4):
                                address6 = TreeNode(
                                    self._input[self._offset : self._offset + 1], self._offset
                                )
                                self._offset = self._offset + 1
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append("[\\d]")
                            if address6 is not FAILURE:
                                elements0.append(address6)
                                address7 = FAILURE
                                chunk5 = None
                                if self._offset < self._input_size:
                                    chunk5 = self._input[self._offset : self._offset + 1]
                                if chunk5 is not None and Grammar.REGEX_17.search(chunk5):
                                    address7 = TreeNode(
                                        self._input[self._offset : self._offset + 1],
                                        self._offset,
                                    )
                                    self._offset = self._offset + 1
                                else:
                                    address7 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append("[\\d]")
                                if address7 is not FAILURE:
                                    elements0.append(address7)
                                    address8 = FAILURE
                                    chunk6 = None
                                    if self._offset < self._input_size:
                                        chunk6 = self._input[self._offset : self._offset + 1]
                                    if chunk6 is not None and Grammar.REGEX_18.search(chunk6):
                                        address8 = TreeNode(
                                            self._input[self._offset : self._offset + 1],
                                            self._offset,
                                        )
                                        self._offset = self._offset + 1
                                    else:
                                        address8 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append("[\\d]")
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
            address0 = TreeNode5(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["varwind"][index0] = (address0, self._offset)
        return address0

    def _read_vis(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["vis"].get(index0)
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
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 1]
            if chunk0 is not None and Grammar.REGEX_19.search(chunk0):
                address3 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address3 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append("[\\d]")
            if address3 is not FAILURE:
                elements1.append(address3)
                address4 = FAILURE
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset : self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_20.search(chunk1):
                    address4 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address4 is not FAILURE:
                    elements1.append(address4)
                    address5 = FAILURE
                    chunk2 = None
                    if self._offset < self._input_size:
                        chunk2 = self._input[self._offset : self._offset + 1]
                    if chunk2 is not None and Grammar.REGEX_21.search(chunk2):
                        address5 = TreeNode(
                            self._input[self._offset : self._offset + 1], self._offset
                        )
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append("[\\d]")
                    if address5 is not FAILURE:
                        elements1.append(address5)
                        address6 = FAILURE
                        chunk3 = None
                        if self._offset < self._input_size:
                            chunk3 = self._input[self._offset : self._offset + 1]
                        if chunk3 is not None and Grammar.REGEX_22.search(chunk3):
                            address6 = TreeNode(
                                self._input[self._offset : self._offset + 1], self._offset
                            )
                            self._offset = self._offset + 1
                        else:
                            address6 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append("[\\d]")
                        if address6 is not FAILURE:
                            elements1.append(address6)
                            address7 = FAILURE
                            index5 = self._offset
                            chunk4 = None
                            if self._offset < self._input_size:
                                chunk4 = self._input[self._offset : self._offset + 3]
                            if chunk4 == "NDV":
                                address7 = TreeNode(
                                    self._input[self._offset : self._offset + 3], self._offset
                                )
                                self._offset = self._offset + 3
                            else:
                                address7 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('"NDV"')
                            if address7 is FAILURE:
                                address7 = TreeNode(self._input[index5:index5], index5)
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
                address2 = TreeNode(self._input[index4 : self._offset], index4, elements1)
                self._offset = self._offset
            if address2 is FAILURE:
                self._offset = index3
                index6, elements2 = self._offset, []
                address8 = FAILURE
                chunk5 = None
                if self._offset < self._input_size:
                    chunk5 = self._input[self._offset : self._offset + 1]
                if chunk5 is not None and Grammar.REGEX_23.search(chunk5):
                    address8 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address8 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address8 is not FAILURE:
                    elements2.append(address8)
                    address9 = FAILURE
                    index7 = self._offset
                    index8 = self._offset
                    chunk6 = None
                    if self._offset < self._input_size:
                        chunk6 = self._input[self._offset : self._offset + 1]
                    if chunk6 is not None and Grammar.REGEX_24.search(chunk6):
                        address9 = TreeNode(
                            self._input[self._offset : self._offset + 1], self._offset
                        )
                        self._offset = self._offset + 1
                    else:
                        address9 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append("[\\d]")
                    if address9 is FAILURE:
                        self._offset = index8
                        index9, elements3 = self._offset, []
                        address10 = FAILURE
                        index10 = self._offset
                        index11, elements4 = self._offset, []
                        address11 = FAILURE
                        chunk7 = None
                        if self._offset < self._input_size:
                            chunk7 = self._input[self._offset : self._offset + 1]
                        if chunk7 == " ":
                            address11 = TreeNode(
                                self._input[self._offset : self._offset + 1], self._offset
                            )
                            self._offset = self._offset + 1
                        else:
                            address11 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('" "')
                        if address11 is not FAILURE:
                            elements4.append(address11)
                            address12 = FAILURE
                            chunk8 = None
                            if self._offset < self._input_size:
                                chunk8 = self._input[self._offset : self._offset + 1]
                            if chunk8 is not None and Grammar.REGEX_25.search(chunk8):
                                address12 = TreeNode(
                                    self._input[self._offset : self._offset + 1], self._offset
                                )
                                self._offset = self._offset + 1
                            else:
                                address12 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append("[\\d]")
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
                            address10 = TreeNode(
                                self._input[index11 : self._offset], index11, elements4
                            )
                            self._offset = self._offset
                        if address10 is FAILURE:
                            address10 = TreeNode(self._input[index10:index10], index10)
                            self._offset = index10
                        if address10 is not FAILURE:
                            elements3.append(address10)
                            address13 = FAILURE
                            chunk9 = None
                            if self._offset < self._input_size:
                                chunk9 = self._input[self._offset : self._offset + 1]
                            if chunk9 == "/":
                                address13 = TreeNode(
                                    self._input[self._offset : self._offset + 1], self._offset
                                )
                                self._offset = self._offset + 1
                            else:
                                address13 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('"/"')
                            if address13 is not FAILURE:
                                elements3.append(address13)
                                address14 = FAILURE
                                chunk10 = None
                                if self._offset < self._input_size:
                                    chunk10 = self._input[self._offset : self._offset + 1]
                                if chunk10 is not None and Grammar.REGEX_26.search(chunk10):
                                    address14 = TreeNode(
                                        self._input[self._offset : self._offset + 1],
                                        self._offset,
                                    )
                                    self._offset = self._offset + 1
                                else:
                                    address14 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append("[\\d]")
                                if address14 is not FAILURE:
                                    elements3.append(address14)
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
                            address9 = TreeNode(
                                self._input[index9 : self._offset], index9, elements3
                            )
                            self._offset = self._offset
                        if address9 is FAILURE:
                            self._offset = index8
                    if address9 is FAILURE:
                        address9 = TreeNode(self._input[index7:index7], index7)
                        self._offset = index7
                    if address9 is not FAILURE:
                        elements2.append(address9)
                        address15 = FAILURE
                        chunk11 = None
                        if self._offset < self._input_size:
                            chunk11 = self._input[self._offset : self._offset + 2]
                        if chunk11 == "SM":
                            address15 = TreeNode(
                                self._input[self._offset : self._offset + 2], self._offset
                            )
                            self._offset = self._offset + 2
                        else:
                            address15 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('"SM"')
                        if address15 is not FAILURE:
                            elements2.append(address15)
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
                    address2 = TreeNode(self._input[index6 : self._offset], index6, elements2)
                    self._offset = self._offset
                if address2 is FAILURE:
                    self._offset = index3
                    chunk12 = None
                    if self._offset < self._input_size:
                        chunk12 = self._input[self._offset : self._offset + 5]
                    if chunk12 == "CAVOK":
                        address2 = TreeNode(
                            self._input[self._offset : self._offset + 5], self._offset
                        )
                        self._offset = self._offset + 5
                    else:
                        address2 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"CAVOK"')
                    if address2 is FAILURE:
                        self._offset = index3
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
            address0 = TreeNode6(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["vis"][index0] = (address0, self._offset)
        return address0

    def _read_run(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["run"].get(index0)
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
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 1]
            if chunk0 == "R":
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"R"')
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index3 = self._offset
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset : self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_27.search(chunk1):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[LRC]")
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index3:index3], index3)
                    self._offset = index3
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk2 = None
                    if self._offset < self._input_size:
                        chunk2 = self._input[self._offset : self._offset + 1]
                    if chunk2 is not None and Grammar.REGEX_28.search(chunk2):
                        address4 = TreeNode(
                            self._input[self._offset : self._offset + 1], self._offset
                        )
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append("[\\d]")
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        chunk3 = None
                        if self._offset < self._input_size:
                            chunk3 = self._input[self._offset : self._offset + 1]
                        if chunk3 is not None and Grammar.REGEX_29.search(chunk3):
                            address5 = TreeNode(
                                self._input[self._offset : self._offset + 1], self._offset
                            )
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append("[\\d]")
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            index4 = self._offset
                            chunk4 = None
                            if self._offset < self._input_size:
                                chunk4 = self._input[self._offset : self._offset + 1]
                            if chunk4 is not None and Grammar.REGEX_30.search(chunk4):
                                address6 = TreeNode(
                                    self._input[self._offset : self._offset + 1], self._offset
                                )
                                self._offset = self._offset + 1
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append("[LRC]")
                            if address6 is FAILURE:
                                address6 = TreeNode(self._input[index4:index4], index4)
                                self._offset = index4
                            if address6 is not FAILURE:
                                elements0.append(address6)
                                address7 = FAILURE
                                chunk5 = None
                                if self._offset < self._input_size:
                                    chunk5 = self._input[self._offset : self._offset + 1]
                                if chunk5 == "/":
                                    address7 = TreeNode(
                                        self._input[self._offset : self._offset + 1],
                                        self._offset,
                                    )
                                    self._offset = self._offset + 1
                                else:
                                    address7 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('"/"')
                                if address7 is not FAILURE:
                                    elements0.append(address7)
                                    address8 = FAILURE
                                    index5 = self._offset
                                    index6, elements1 = self._offset, []
                                    address9 = FAILURE
                                    chunk6 = None
                                    if self._offset < self._input_size:
                                        chunk6 = self._input[self._offset : self._offset + 1]
                                    if chunk6 is not None and Grammar.REGEX_31.search(chunk6):
                                        address9 = TreeNode(
                                            self._input[self._offset : self._offset + 1],
                                            self._offset,
                                        )
                                        self._offset = self._offset + 1
                                    else:
                                        address9 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append("[\\d]")
                                    if address9 is not FAILURE:
                                        elements1.append(address9)
                                        address10 = FAILURE
                                        chunk7 = None
                                        if self._offset < self._input_size:
                                            chunk7 = self._input[
                                                self._offset : self._offset + 1
                                            ]
                                        if chunk7 is not None and Grammar.REGEX_32.search(
                                            chunk7
                                        ):
                                            address10 = TreeNode(
                                                self._input[self._offset : self._offset + 1],
                                                self._offset,
                                            )
                                            self._offset = self._offset + 1
                                        else:
                                            address10 = FAILURE
                                            if self._offset > self._failure:
                                                self._failure = self._offset
                                                self._expected = []
                                            if self._offset == self._failure:
                                                self._expected.append("[\\d]")
                                        if address10 is not FAILURE:
                                            elements1.append(address10)
                                            address11 = FAILURE
                                            chunk8 = None
                                            if self._offset < self._input_size:
                                                chunk8 = self._input[
                                                    self._offset : self._offset + 1
                                                ]
                                            if chunk8 is not None and Grammar.REGEX_33.search(
                                                chunk8
                                            ):
                                                address11 = TreeNode(
                                                    self._input[
                                                        self._offset : self._offset + 1
                                                    ],
                                                    self._offset,
                                                )
                                                self._offset = self._offset + 1
                                            else:
                                                address11 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append("[\\d]")
                                            if address11 is not FAILURE:
                                                elements1.append(address11)
                                                address12 = FAILURE
                                                chunk9 = None
                                                if self._offset < self._input_size:
                                                    chunk9 = self._input[
                                                        self._offset : self._offset + 1
                                                    ]
                                                if (
                                                    chunk9 is not None
                                                    and Grammar.REGEX_34.search(chunk9)
                                                ):
                                                    address12 = TreeNode(
                                                        self._input[
                                                            self._offset : self._offset + 1
                                                        ],
                                                        self._offset,
                                                    )
                                                    self._offset = self._offset + 1
                                                else:
                                                    address12 = FAILURE
                                                    if self._offset > self._failure:
                                                        self._failure = self._offset
                                                        self._expected = []
                                                    if self._offset == self._failure:
                                                        self._expected.append("[\\d]")
                                                if address12 is not FAILURE:
                                                    elements1.append(address12)
                                                    address13 = FAILURE
                                                    chunk10 = None
                                                    if self._offset < self._input_size:
                                                        chunk10 = self._input[
                                                            self._offset : self._offset + 1
                                                        ]
                                                    if chunk10 == "V":
                                                        address13 = TreeNode(
                                                            self._input[
                                                                self._offset : self._offset + 1
                                                            ],
                                                            self._offset,
                                                        )
                                                        self._offset = self._offset + 1
                                                    else:
                                                        address13 = FAILURE
                                                        if self._offset > self._failure:
                                                            self._failure = self._offset
                                                            self._expected = []
                                                        if self._offset == self._failure:
                                                            self._expected.append('"V"')
                                                    if address13 is not FAILURE:
                                                        elements1.append(address13)
                                                    else:
                                                        elements1 = None
                                                        self._offset = index6
                                                else:
                                                    elements1 = None
                                                    self._offset = index6
                                            else:
                                                elements1 = None
                                                self._offset = index6
                                        else:
                                            elements1 = None
                                            self._offset = index6
                                    else:
                                        elements1 = None
                                        self._offset = index6
                                    if elements1 is None:
                                        address8 = FAILURE
                                    else:
                                        address8 = TreeNode(
                                            self._input[index6 : self._offset],
                                            index6,
                                            elements1,
                                        )
                                        self._offset = self._offset
                                    if address8 is FAILURE:
                                        address8 = TreeNode(self._input[index5:index5], index5)
                                        self._offset = index5
                                    if address8 is not FAILURE:
                                        elements0.append(address8)
                                        address14 = FAILURE
                                        index7 = self._offset
                                        chunk11 = None
                                        if self._offset < self._input_size:
                                            chunk11 = self._input[
                                                self._offset : self._offset + 1
                                            ]
                                        if chunk11 is not None and Grammar.REGEX_35.search(
                                            chunk11
                                        ):
                                            address14 = TreeNode(
                                                self._input[self._offset : self._offset + 1],
                                                self._offset,
                                            )
                                            self._offset = self._offset + 1
                                        else:
                                            address14 = FAILURE
                                            if self._offset > self._failure:
                                                self._failure = self._offset
                                                self._expected = []
                                            if self._offset == self._failure:
                                                self._expected.append('["M" / "P"]')
                                        if address14 is FAILURE:
                                            address14 = TreeNode(
                                                self._input[index7:index7], index7
                                            )
                                            self._offset = index7
                                        if address14 is not FAILURE:
                                            elements0.append(address14)
                                            address15 = FAILURE
                                            chunk12 = None
                                            if self._offset < self._input_size:
                                                chunk12 = self._input[
                                                    self._offset : self._offset + 1
                                                ]
                                            if (
                                                chunk12 is not None
                                                and Grammar.REGEX_36.search(chunk12)
                                            ):
                                                address15 = TreeNode(
                                                    self._input[
                                                        self._offset : self._offset + 1
                                                    ],
                                                    self._offset,
                                                )
                                                self._offset = self._offset + 1
                                            else:
                                                address15 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append("[\\d]")
                                            if address15 is not FAILURE:
                                                elements0.append(address15)
                                                address16 = FAILURE
                                                chunk13 = None
                                                if self._offset < self._input_size:
                                                    chunk13 = self._input[
                                                        self._offset : self._offset + 1
                                                    ]
                                                if (
                                                    chunk13 is not None
                                                    and Grammar.REGEX_37.search(chunk13)
                                                ):
                                                    address16 = TreeNode(
                                                        self._input[
                                                            self._offset : self._offset + 1
                                                        ],
                                                        self._offset,
                                                    )
                                                    self._offset = self._offset + 1
                                                else:
                                                    address16 = FAILURE
                                                    if self._offset > self._failure:
                                                        self._failure = self._offset
                                                        self._expected = []
                                                    if self._offset == self._failure:
                                                        self._expected.append("[\\d]")
                                                if address16 is not FAILURE:
                                                    elements0.append(address16)
                                                    address17 = FAILURE
                                                    chunk14 = None
                                                    if self._offset < self._input_size:
                                                        chunk14 = self._input[
                                                            self._offset : self._offset + 1
                                                        ]
                                                    if (
                                                        chunk14 is not None
                                                        and Grammar.REGEX_38.search(chunk14)
                                                    ):
                                                        address17 = TreeNode(
                                                            self._input[
                                                                self._offset : self._offset + 1
                                                            ],
                                                            self._offset,
                                                        )
                                                        self._offset = self._offset + 1
                                                    else:
                                                        address17 = FAILURE
                                                        if self._offset > self._failure:
                                                            self._failure = self._offset
                                                            self._expected = []
                                                        if self._offset == self._failure:
                                                            self._expected.append("[\\d]")
                                                    if address17 is not FAILURE:
                                                        elements0.append(address17)
                                                        address18 = FAILURE
                                                        chunk15 = None
                                                        if self._offset < self._input_size:
                                                            chunk15 = self._input[
                                                                self._offset : self._offset + 1
                                                            ]
                                                        if (
                                                            chunk15 is not None
                                                            and Grammar.REGEX_39.search(
                                                                chunk15
                                                            )
                                                        ):
                                                            address18 = TreeNode(
                                                                self._input[
                                                                    self._offset : self._offset
                                                                    + 1
                                                                ],
                                                                self._offset,
                                                            )
                                                            self._offset = self._offset + 1
                                                        else:
                                                            address18 = FAILURE
                                                            if self._offset > self._failure:
                                                                self._failure = self._offset
                                                                self._expected = []
                                                            if self._offset == self._failure:
                                                                self._expected.append("[\\d]")
                                                        if address18 is not FAILURE:
                                                            elements0.append(address18)
                                                            address19 = FAILURE
                                                            chunk16 = None
                                                            if self._offset < self._input_size:
                                                                chunk16 = self._input[
                                                                    self._offset : self._offset
                                                                    + 2
                                                                ]
                                                            if chunk16 == "FT":
                                                                address19 = TreeNode(
                                                                    self._input[
                                                                        self._offset : self._offset
                                                                        + 2
                                                                    ],
                                                                    self._offset,
                                                                )
                                                                self._offset = self._offset + 2
                                                            else:
                                                                address19 = FAILURE
                                                                if (
                                                                    self._offset
                                                                    > self._failure
                                                                ):
                                                                    self._failure = (
                                                                        self._offset
                                                                    )
                                                                    self._expected = []
                                                                if (
                                                                    self._offset
                                                                    == self._failure
                                                                ):
                                                                    self._expected.append(
                                                                        '"FT"'
                                                                    )
                                                            if address19 is not FAILURE:
                                                                elements0.append(address19)
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
            address0 = TreeNode7(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["run"][index0] = (address0, self._offset)
        return address0

    def _read_curwx(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["curwx"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        remaining0, index2, elements0, address1 = 0, self._offset, [], True
        while address1 is not FAILURE:
            index3, elements1 = self._offset, []
            address2 = FAILURE
            address2 = self._read_sep()
            if address2 is not FAILURE:
                elements1.append(address2)
                address3 = FAILURE
                address3 = self._read_wx()
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
                address1 = TreeNode8(self._input[index3 : self._offset], index3, elements1)
                self._offset = self._offset
            if address1 is not FAILURE:
                elements0.append(address1)
                remaining0 -= 1
        if remaining0 <= 0:
            address0 = TreeNode(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        else:
            address0 = FAILURE
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["curwx"][index0] = (address0, self._offset)
        return address0

    def _read_wx(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["wx"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        index3 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset : self._offset + 1]
        if chunk0 is not None and Grammar.REGEX_40.search(chunk0):
            address1 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append("[-+]")
        if address1 is FAILURE:
            self._offset = index3
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset : self._offset + 2]
            if chunk1 == "VC":
                address1 = TreeNode(self._input[self._offset : self._offset + 2], self._offset)
                self._offset = self._offset + 2
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"VC"')
            if address1 is FAILURE:
                self._offset = index3
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2)
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index4 = self._offset
            chunk2 = None
            if self._offset < self._input_size:
                chunk2 = self._input[self._offset : self._offset + 2]
            if chunk2 == "MI":
                address2 = TreeNode(self._input[self._offset : self._offset + 2], self._offset)
                self._offset = self._offset + 2
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"MI"')
            if address2 is FAILURE:
                self._offset = index4
                chunk3 = None
                if self._offset < self._input_size:
                    chunk3 = self._input[self._offset : self._offset + 2]
                if chunk3 == "PR":
                    address2 = TreeNode(
                        self._input[self._offset : self._offset + 2], self._offset
                    )
                    self._offset = self._offset + 2
                else:
                    address2 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"PR"')
                if address2 is FAILURE:
                    self._offset = index4
                    chunk4 = None
                    if self._offset < self._input_size:
                        chunk4 = self._input[self._offset : self._offset + 2]
                    if chunk4 == "DR":
                        address2 = TreeNode(
                            self._input[self._offset : self._offset + 2], self._offset
                        )
                        self._offset = self._offset + 2
                    else:
                        address2 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"DR"')
                    if address2 is FAILURE:
                        self._offset = index4
                        chunk5 = None
                        if self._offset < self._input_size:
                            chunk5 = self._input[self._offset : self._offset + 2]
                        if chunk5 == "BL":
                            address2 = TreeNode(
                                self._input[self._offset : self._offset + 2], self._offset
                            )
                            self._offset = self._offset + 2
                        else:
                            address2 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('"BL"')
                        if address2 is FAILURE:
                            self._offset = index4
                            chunk6 = None
                            if self._offset < self._input_size:
                                chunk6 = self._input[self._offset : self._offset + 2]
                            if chunk6 == "SH":
                                address2 = TreeNode(
                                    self._input[self._offset : self._offset + 2], self._offset
                                )
                                self._offset = self._offset + 2
                            else:
                                address2 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('"SH"')
                            if address2 is FAILURE:
                                self._offset = index4
                                chunk7 = None
                                if self._offset < self._input_size:
                                    chunk7 = self._input[self._offset : self._offset + 2]
                                if chunk7 == "TS":
                                    address2 = TreeNode(
                                        self._input[self._offset : self._offset + 2],
                                        self._offset,
                                    )
                                    self._offset = self._offset + 2
                                else:
                                    address2 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('"TS"')
                                if address2 is FAILURE:
                                    self._offset = index4
                                    chunk8 = None
                                    if self._offset < self._input_size:
                                        chunk8 = self._input[self._offset : self._offset + 2]
                                    if chunk8 == "FG":
                                        address2 = TreeNode(
                                            self._input[self._offset : self._offset + 2],
                                            self._offset,
                                        )
                                        self._offset = self._offset + 2
                                    else:
                                        address2 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append('"FG"')
                                    if address2 is FAILURE:
                                        self._offset = index4
                                        chunk9 = None
                                        if self._offset < self._input_size:
                                            chunk9 = self._input[
                                                self._offset : self._offset + 2
                                            ]
                                        if chunk9 == "TS":
                                            address2 = TreeNode(
                                                self._input[self._offset : self._offset + 2],
                                                self._offset,
                                            )
                                            self._offset = self._offset + 2
                                        else:
                                            address2 = FAILURE
                                            if self._offset > self._failure:
                                                self._failure = self._offset
                                                self._expected = []
                                            if self._offset == self._failure:
                                                self._expected.append('"TS"')
                                        if address2 is FAILURE:
                                            self._offset = index4
                                            chunk10 = None
                                            if self._offset < self._input_size:
                                                chunk10 = self._input[
                                                    self._offset : self._offset + 2
                                                ]
                                            if chunk10 == "FZ":
                                                address2 = TreeNode(
                                                    self._input[
                                                        self._offset : self._offset + 2
                                                    ],
                                                    self._offset,
                                                )
                                                self._offset = self._offset + 2
                                            else:
                                                address2 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append('"FZ"')
                                            if address2 is FAILURE:
                                                self._offset = index4
                                                chunk11 = None
                                                if self._offset < self._input_size:
                                                    chunk11 = self._input[
                                                        self._offset : self._offset + 2
                                                    ]
                                                if chunk11 == "RA":
                                                    address2 = TreeNode(
                                                        self._input[
                                                            self._offset : self._offset + 2
                                                        ],
                                                        self._offset,
                                                    )
                                                    self._offset = self._offset + 2
                                                else:
                                                    address2 = FAILURE
                                                    if self._offset > self._failure:
                                                        self._failure = self._offset
                                                        self._expected = []
                                                    if self._offset == self._failure:
                                                        self._expected.append('"RA"')
                                                if address2 is FAILURE:
                                                    self._offset = index4
                                                    chunk12 = None
                                                    if self._offset < self._input_size:
                                                        chunk12 = self._input[
                                                            self._offset : self._offset + 2
                                                        ]
                                                    if chunk12 == "BR":
                                                        address2 = TreeNode(
                                                            self._input[
                                                                self._offset : self._offset + 2
                                                            ],
                                                            self._offset,
                                                        )
                                                        self._offset = self._offset + 2
                                                    else:
                                                        address2 = FAILURE
                                                        if self._offset > self._failure:
                                                            self._failure = self._offset
                                                            self._expected = []
                                                        if self._offset == self._failure:
                                                            self._expected.append('"BR"')
                                                    if address2 is FAILURE:
                                                        self._offset = index4
                                                        chunk13 = None
                                                        if self._offset < self._input_size:
                                                            chunk13 = self._input[
                                                                self._offset : self._offset + 2
                                                            ]
                                                        if chunk13 == "HZ":
                                                            address2 = TreeNode(
                                                                self._input[
                                                                    self._offset : self._offset
                                                                    + 2
                                                                ],
                                                                self._offset,
                                                            )
                                                            self._offset = self._offset + 2
                                                        else:
                                                            address2 = FAILURE
                                                            if self._offset > self._failure:
                                                                self._failure = self._offset
                                                                self._expected = []
                                                            if self._offset == self._failure:
                                                                self._expected.append('"HZ"')
                                                        if address2 is FAILURE:
                                                            self._offset = index4
                                                            chunk14 = None
                                                            if self._offset < self._input_size:
                                                                chunk14 = self._input[
                                                                    self._offset : self._offset
                                                                    + 2
                                                                ]
                                                            if chunk14 == "SN":
                                                                address2 = TreeNode(
                                                                    self._input[
                                                                        self._offset : self._offset
                                                                        + 2
                                                                    ],
                                                                    self._offset,
                                                                )
                                                                self._offset = self._offset + 2
                                                            else:
                                                                address2 = FAILURE
                                                                if (
                                                                    self._offset
                                                                    > self._failure
                                                                ):
                                                                    self._failure = (
                                                                        self._offset
                                                                    )
                                                                    self._expected = []
                                                                if (
                                                                    self._offset
                                                                    == self._failure
                                                                ):
                                                                    self._expected.append(
                                                                        '"SN"'
                                                                    )
                                                            if address2 is FAILURE:
                                                                self._offset = index4
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index5 = self._offset
                chunk15 = None
                if self._offset < self._input_size:
                    chunk15 = self._input[self._offset : self._offset + 1]
                if chunk15 is not None and Grammar.REGEX_41.search(chunk15):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[-+]")
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index5:index5], index5)
                    self._offset = index5
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    index6 = self._offset
                    index7 = self._offset
                    chunk16 = None
                    if self._offset < self._input_size:
                        chunk16 = self._input[self._offset : self._offset + 2]
                    if chunk16 == "RA":
                        address4 = TreeNode(
                            self._input[self._offset : self._offset + 2], self._offset
                        )
                        self._offset = self._offset + 2
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"RA"')
                    if address4 is FAILURE:
                        self._offset = index7
                        chunk17 = None
                        if self._offset < self._input_size:
                            chunk17 = self._input[self._offset : self._offset + 2]
                        if chunk17 == "BR":
                            address4 = TreeNode(
                                self._input[self._offset : self._offset + 2], self._offset
                            )
                            self._offset = self._offset + 2
                        else:
                            address4 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('"BR"')
                        if address4 is FAILURE:
                            self._offset = index7
                            chunk18 = None
                            if self._offset < self._input_size:
                                chunk18 = self._input[self._offset : self._offset + 2]
                            if chunk18 == "DZ":
                                address4 = TreeNode(
                                    self._input[self._offset : self._offset + 2], self._offset
                                )
                                self._offset = self._offset + 2
                            else:
                                address4 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('"DZ"')
                            if address4 is FAILURE:
                                self._offset = index7
                                chunk19 = None
                                if self._offset < self._input_size:
                                    chunk19 = self._input[self._offset : self._offset + 2]
                                if chunk19 == "FG":
                                    address4 = TreeNode(
                                        self._input[self._offset : self._offset + 2],
                                        self._offset,
                                    )
                                    self._offset = self._offset + 2
                                else:
                                    address4 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('"FG"')
                                if address4 is FAILURE:
                                    self._offset = index7
                                    chunk20 = None
                                    if self._offset < self._input_size:
                                        chunk20 = self._input[self._offset : self._offset + 2]
                                    if chunk20 == "FU":
                                        address4 = TreeNode(
                                            self._input[self._offset : self._offset + 2],
                                            self._offset,
                                        )
                                        self._offset = self._offset + 2
                                    else:
                                        address4 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append('"FU"')
                                    if address4 is FAILURE:
                                        self._offset = index7
                                        chunk21 = None
                                        if self._offset < self._input_size:
                                            chunk21 = self._input[
                                                self._offset : self._offset + 2
                                            ]
                                        if chunk21 == "VA":
                                            address4 = TreeNode(
                                                self._input[self._offset : self._offset + 2],
                                                self._offset,
                                            )
                                            self._offset = self._offset + 2
                                        else:
                                            address4 = FAILURE
                                            if self._offset > self._failure:
                                                self._failure = self._offset
                                                self._expected = []
                                            if self._offset == self._failure:
                                                self._expected.append('"VA"')
                                        if address4 is FAILURE:
                                            self._offset = index7
                                            chunk22 = None
                                            if self._offset < self._input_size:
                                                chunk22 = self._input[
                                                    self._offset : self._offset + 2
                                                ]
                                            if chunk22 == "DU":
                                                address4 = TreeNode(
                                                    self._input[
                                                        self._offset : self._offset + 2
                                                    ],
                                                    self._offset,
                                                )
                                                self._offset = self._offset + 2
                                            else:
                                                address4 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append('"DU"')
                                            if address4 is FAILURE:
                                                self._offset = index7
                                                chunk23 = None
                                                if self._offset < self._input_size:
                                                    chunk23 = self._input[
                                                        self._offset : self._offset + 2
                                                    ]
                                                if chunk23 == "SA":
                                                    address4 = TreeNode(
                                                        self._input[
                                                            self._offset : self._offset + 2
                                                        ],
                                                        self._offset,
                                                    )
                                                    self._offset = self._offset + 2
                                                else:
                                                    address4 = FAILURE
                                                    if self._offset > self._failure:
                                                        self._failure = self._offset
                                                        self._expected = []
                                                    if self._offset == self._failure:
                                                        self._expected.append('"SA"')
                                                if address4 is FAILURE:
                                                    self._offset = index7
                                                    chunk24 = None
                                                    if self._offset < self._input_size:
                                                        chunk24 = self._input[
                                                            self._offset : self._offset + 2
                                                        ]
                                                    if chunk24 == "SA":
                                                        address4 = TreeNode(
                                                            self._input[
                                                                self._offset : self._offset + 2
                                                            ],
                                                            self._offset,
                                                        )
                                                        self._offset = self._offset + 2
                                                    else:
                                                        address4 = FAILURE
                                                        if self._offset > self._failure:
                                                            self._failure = self._offset
                                                            self._expected = []
                                                        if self._offset == self._failure:
                                                            self._expected.append('"SA"')
                                                    if address4 is FAILURE:
                                                        self._offset = index7
                                                        chunk25 = None
                                                        if self._offset < self._input_size:
                                                            chunk25 = self._input[
                                                                self._offset : self._offset + 2
                                                            ]
                                                        if chunk25 == "HZ":
                                                            address4 = TreeNode(
                                                                self._input[
                                                                    self._offset : self._offset
                                                                    + 2
                                                                ],
                                                                self._offset,
                                                            )
                                                            self._offset = self._offset + 2
                                                        else:
                                                            address4 = FAILURE
                                                            if self._offset > self._failure:
                                                                self._failure = self._offset
                                                                self._expected = []
                                                            if self._offset == self._failure:
                                                                self._expected.append('"HZ"')
                                                        if address4 is FAILURE:
                                                            self._offset = index7
                                                            chunk26 = None
                                                            if self._offset < self._input_size:
                                                                chunk26 = self._input[
                                                                    self._offset : self._offset
                                                                    + 2
                                                                ]
                                                            if chunk26 == "PY":
                                                                address4 = TreeNode(
                                                                    self._input[
                                                                        self._offset : self._offset
                                                                        + 2
                                                                    ],
                                                                    self._offset,
                                                                )
                                                                self._offset = self._offset + 2
                                                            else:
                                                                address4 = FAILURE
                                                                if (
                                                                    self._offset
                                                                    > self._failure
                                                                ):
                                                                    self._failure = (
                                                                        self._offset
                                                                    )
                                                                    self._expected = []
                                                                if (
                                                                    self._offset
                                                                    == self._failure
                                                                ):
                                                                    self._expected.append(
                                                                        '"PY"'
                                                                    )
                                                            if address4 is FAILURE:
                                                                self._offset = index7
                    if address4 is FAILURE:
                        address4 = TreeNode(self._input[index6:index6], index6)
                        self._offset = index6
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
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["wx"][index0] = (address0, self._offset)
        return address0

    def _read_skyc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["skyc"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        remaining0, index2, elements0, address1 = 0, self._offset, [], True
        while address1 is not FAILURE:
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
                address1 = TreeNode9(self._input[index3 : self._offset], index3, elements1)
                self._offset = self._offset
            if address1 is not FAILURE:
                elements0.append(address1)
                remaining0 -= 1
        if remaining0 <= 0:
            address0 = TreeNode(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        else:
            address0 = FAILURE
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["skyc"][index0] = (address0, self._offset)
        return address0

    def _read_cover(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["cover"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset : self._offset + 3]
        if chunk0 == "FEW":
            address1 = TreeNode(self._input[self._offset : self._offset + 3], self._offset)
            self._offset = self._offset + 3
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"FEW"')
        if address1 is FAILURE:
            self._offset = index3
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset : self._offset + 3]
            if chunk1 == "SCT":
                address1 = TreeNode(self._input[self._offset : self._offset + 3], self._offset)
                self._offset = self._offset + 3
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"SCT"')
            if address1 is FAILURE:
                self._offset = index3
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset : self._offset + 3]
                if chunk2 == "BKN":
                    address1 = TreeNode(
                        self._input[self._offset : self._offset + 3], self._offset
                    )
                    self._offset = self._offset + 3
                else:
                    address1 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"BKN"')
                if address1 is FAILURE:
                    self._offset = index3
                    chunk3 = None
                    if self._offset < self._input_size:
                        chunk3 = self._input[self._offset : self._offset + 3]
                    if chunk3 == "OVC":
                        address1 = TreeNode(
                            self._input[self._offset : self._offset + 3], self._offset
                        )
                        self._offset = self._offset + 3
                    else:
                        address1 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"OVC"')
                    if address1 is FAILURE:
                        self._offset = index3
                        chunk4 = None
                        if self._offset < self._input_size:
                            chunk4 = self._input[self._offset : self._offset + 2]
                        if chunk4 == "VV":
                            address1 = TreeNode(
                                self._input[self._offset : self._offset + 2], self._offset
                            )
                            self._offset = self._offset + 2
                        else:
                            address1 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('"VV"')
                        if address1 is FAILURE:
                            self._offset = index3
                            chunk5 = None
                            if self._offset < self._input_size:
                                chunk5 = self._input[self._offset : self._offset + 3]
                            if chunk5 == "///":
                                address1 = TreeNode(
                                    self._input[self._offset : self._offset + 3], self._offset
                                )
                                self._offset = self._offset + 3
                            else:
                                address1 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('"///"')
                            if address1 is FAILURE:
                                self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index4 = self._offset
            remaining0, index5, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                chunk6 = None
                if self._offset < self._input_size:
                    chunk6 = self._input[self._offset : self._offset + 1]
                if chunk6 is not None and Grammar.REGEX_42.search(chunk6):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index5 : self._offset], index5, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index4:index4], index4)
                self._offset = index4
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                index6 = self._offset
                index7 = self._offset
                chunk7 = None
                if self._offset < self._input_size:
                    chunk7 = self._input[self._offset : self._offset + 3]
                if chunk7 == "TCU":
                    address4 = TreeNode(
                        self._input[self._offset : self._offset + 3], self._offset
                    )
                    self._offset = self._offset + 3
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"TCU"')
                if address4 is FAILURE:
                    self._offset = index7
                    chunk8 = None
                    if self._offset < self._input_size:
                        chunk8 = self._input[self._offset : self._offset + 2]
                    if chunk8 == "CB":
                        address4 = TreeNode(
                            self._input[self._offset : self._offset + 2], self._offset
                        )
                        self._offset = self._offset + 2
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"CB"')
                    if address4 is FAILURE:
                        self._offset = index7
                        chunk9 = None
                        if self._offset < self._input_size:
                            chunk9 = self._input[self._offset : self._offset + 3]
                        if chunk9 == "///":
                            address4 = TreeNode(
                                self._input[self._offset : self._offset + 3], self._offset
                            )
                            self._offset = self._offset + 3
                        else:
                            address4 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('"///"')
                        if address4 is FAILURE:
                            self._offset = index7
                if address4 is FAILURE:
                    address4 = TreeNode(self._input[index6:index6], index6)
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
            address0 = TreeNode(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index1
            index8 = self._offset
            chunk10 = None
            if self._offset < self._input_size:
                chunk10 = self._input[self._offset : self._offset + 3]
            if chunk10 == "CLR":
                address0 = TreeNode(self._input[self._offset : self._offset + 3], self._offset)
                self._offset = self._offset + 3
            else:
                address0 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"CLR"')
            if address0 is FAILURE:
                self._offset = index8
                chunk11 = None
                if self._offset < self._input_size:
                    chunk11 = self._input[self._offset : self._offset + 3]
                if chunk11 == "SKC":
                    address0 = TreeNode(
                        self._input[self._offset : self._offset + 3], self._offset
                    )
                    self._offset = self._offset + 3
                else:
                    address0 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"SKC"')
                if address0 is FAILURE:
                    self._offset = index8
                    chunk12 = None
                    if self._offset < self._input_size:
                        chunk12 = self._input[self._offset : self._offset + 3]
                    if chunk12 == "NSC":
                        address0 = TreeNode(
                            self._input[self._offset : self._offset + 3], self._offset
                        )
                        self._offset = self._offset + 3
                    else:
                        address0 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"NSC"')
                    if address0 is FAILURE:
                        self._offset = index8
                        chunk13 = None
                        if self._offset < self._input_size:
                            chunk13 = self._input[self._offset : self._offset + 3]
                        if chunk13 == "NCD":
                            address0 = TreeNode(
                                self._input[self._offset : self._offset + 3], self._offset
                            )
                            self._offset = self._offset + 3
                        else:
                            address0 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('"NCD"')
                        if address0 is FAILURE:
                            self._offset = index8
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_wx()
                if address0 is FAILURE:
                    self._offset = index1
                    chunk14 = None
                    if self._offset < self._input_size:
                        chunk14 = self._input[self._offset : self._offset + 2]
                    if chunk14 == "//":
                        address0 = TreeNode(
                            self._input[self._offset : self._offset + 2], self._offset
                        )
                        self._offset = self._offset + 2
                    else:
                        address0 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"//"')
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache["cover"][index0] = (address0, self._offset)
        return address0

    def _read_temp_dewp(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["temp_dewp"].get(index0)
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
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 2]
            if chunk0 == "//":
                address2 = TreeNode(self._input[self._offset : self._offset + 2], self._offset)
                self._offset = self._offset + 2
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"//"')
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index3:index3], index3)
                self._offset = index3
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                address3 = self._read_temp()
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset : self._offset + 1]
                    if chunk1 == "/":
                        address4 = TreeNode(
                            self._input[self._offset : self._offset + 1], self._offset
                        )
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"/"')
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        address5 = self._read_dewp()
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            index4 = self._offset
                            chunk2 = None
                            if self._offset < self._input_size:
                                chunk2 = self._input[self._offset : self._offset + 2]
                            if chunk2 == "//":
                                address6 = TreeNode(
                                    self._input[self._offset : self._offset + 2], self._offset
                                )
                                self._offset = self._offset + 2
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('"//"')
                            if address6 is FAILURE:
                                address6 = TreeNode(self._input[index4:index4], index4)
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
            address0 = TreeNode10(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["temp_dewp"][index0] = (address0, self._offset)
        return address0

    def _read_temp(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["temp"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset : self._offset + 1]
        if chunk0 is not None and Grammar.REGEX_43.search(chunk0):
            address1 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append("[M]")
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2)
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset : self._offset + 1]
            if chunk1 is not None and Grammar.REGEX_44.search(chunk1):
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append("[\\d]")
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index3:index3], index3)
                self._offset = index3
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index4 = self._offset
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset : self._offset + 1]
                if chunk2 is not None and Grammar.REGEX_45.search(chunk2):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index4:index4], index4)
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
            address0 = TreeNode(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["temp"][index0] = (address0, self._offset)
        return address0

    def _read_dewp(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["dewp"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset : self._offset + 1]
        if chunk0 is not None and Grammar.REGEX_46.search(chunk0):
            address1 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append("[M]")
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2)
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset : self._offset + 1]
            if chunk1 is not None and Grammar.REGEX_47.search(chunk1):
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append("[\\d]")
            if address2 is FAILURE:
                address2 = TreeNode(self._input[index3:index3], index3)
                self._offset = index3
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                index4 = self._offset
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset : self._offset + 1]
                if chunk2 is not None and Grammar.REGEX_48.search(chunk2):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address3 is FAILURE:
                    address3 = TreeNode(self._input[index4:index4], index4)
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
            address0 = TreeNode(self._input[index1 : self._offset], index1, elements0)
            self._offset = self._offset
        self._cache["dewp"][index0] = (address0, self._offset)
        return address0

    def _read_altim(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["altim"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index3:index3], index3)
            self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 1]
            if chunk0 is not None and Grammar.REGEX_49.search(chunk0):
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('["Q" / "A"]')
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset : self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_50.search(chunk1):
                    address3 = TreeNode(
                        self._input[self._offset : self._offset + 1], self._offset
                    )
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append("[\\d]")
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    chunk2 = None
                    if self._offset < self._input_size:
                        chunk2 = self._input[self._offset : self._offset + 1]
                    if chunk2 is not None and Grammar.REGEX_51.search(chunk2):
                        address4 = TreeNode(
                            self._input[self._offset : self._offset + 1], self._offset
                        )
                        self._offset = self._offset + 1
                    else:
                        address4 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append("[\\d]")
                    if address4 is not FAILURE:
                        elements0.append(address4)
                        address5 = FAILURE
                        chunk3 = None
                        if self._offset < self._input_size:
                            chunk3 = self._input[self._offset : self._offset + 1]
                        if chunk3 is not None and Grammar.REGEX_52.search(chunk3):
                            address5 = TreeNode(
                                self._input[self._offset : self._offset + 1], self._offset
                            )
                            self._offset = self._offset + 1
                        else:
                            address5 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append("[\\d]")
                        if address5 is not FAILURE:
                            elements0.append(address5)
                            address6 = FAILURE
                            chunk4 = None
                            if self._offset < self._input_size:
                                chunk4 = self._input[self._offset : self._offset + 1]
                            if chunk4 is not None and Grammar.REGEX_53.search(chunk4):
                                address6 = TreeNode(
                                    self._input[self._offset : self._offset + 1], self._offset
                                )
                                self._offset = self._offset + 1
                            else:
                                address6 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append("[\\d]")
                            if address6 is not FAILURE:
                                elements0.append(address6)
                                address7 = FAILURE
                                index4 = self._offset
                                chunk5 = None
                                if self._offset < self._input_size:
                                    chunk5 = self._input[self._offset : self._offset + 1]
                                if chunk5 == "=":
                                    address7 = TreeNode(
                                        self._input[self._offset : self._offset + 1],
                                        self._offset,
                                    )
                                    self._offset = self._offset + 1
                                else:
                                    address7 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('"="')
                                if address7 is FAILURE:
                                    address7 = TreeNode(self._input[index4:index4], index4)
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
            address0 = TreeNode(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["altim"][index0] = (address0, self._offset)
        return address0

    def _read_remarks(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["remarks"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index3:index3], index3)
            self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index4 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 3]
            if chunk0 == "RMK":
                address2 = TreeNode(self._input[self._offset : self._offset + 3], self._offset)
                self._offset = self._offset + 3
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"RMK"')
            if address2 is FAILURE:
                self._offset = index4
                remaining0, index5, elements1, address3 = 0, self._offset, [], True
                while address3 is not FAILURE:
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset : self._offset + 5]
                    if chunk1 == "NOSIG":
                        address3 = TreeNode(
                            self._input[self._offset : self._offset + 5], self._offset
                        )
                        self._offset = self._offset + 5
                    else:
                        address3 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('"NOSIG"')
                    if address3 is not FAILURE:
                        elements1.append(address3)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address2 = TreeNode(self._input[index5 : self._offset], index5, elements1)
                    self._offset = self._offset
                else:
                    address2 = FAILURE
                if address2 is FAILURE:
                    self._offset = index4
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                remaining1, index6, elements2, address5 = 0, self._offset, [], True
                while address5 is not FAILURE:
                    if self._offset < self._input_size:
                        address5 = TreeNode(
                            self._input[self._offset : self._offset + 1], self._offset
                        )
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append("<any char>")
                    if address5 is not FAILURE:
                        elements2.append(address5)
                        remaining1 -= 1
                if remaining1 <= 0:
                    address4 = TreeNode(self._input[index6 : self._offset], index6, elements2)
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
            address0 = TreeNode(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["remarks"][index0] = (address0, self._offset)
        return address0

    def _read_end(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache["end"].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        index3 = self._offset
        address1 = self._read_sep()
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index3:index3], index3)
            self._offset = index3
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset : self._offset + 1]
            if chunk0 == "=":
                address2 = TreeNode(self._input[self._offset : self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"="')
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
            address0 = TreeNode(self._input[index2 : self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            address0 = TreeNode(self._input[index1:index1], index1)
            self._offset = index1
        self._cache["end"][index0] = (address0, self._offset)
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
            self._expected.append("<EOF>")
        raise ParseError(format_error(self._input, self._failure, self._expected))


def format_error(input, offset, expected):
    lines, line_no, position = input.split("\n"), 0, 0
    while position <= offset:
        position += len(lines[line_no]) + 1
        line_no += 1
    message, line = (
        "Line " + str(line_no) + ": expected " + ", ".join(expected) + "\n",
        lines[line_no - 1],
    )
    message += line + "\n"
    position -= len(line) + 1
    message += " " * (offset - position)
    return message + "^"


def parse(input, actions=None, types=None):
    parser = Parser(input, actions, types)
    return parser.parse()
