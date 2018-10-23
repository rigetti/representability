import copy


class DualBasisElement(object):
    """
    Define an element of the dual basis that will satisfy the following equation:

    Ax + b = c

    Dual of the tensors are linear operators on the tensor space producing
    scalar values.

    primal_elements is a list of tensor object names,--e_{i}

    primal_coeffs is a list of coefficients associated with the primal e_{i}

    dual_scalar is the coefficient for the inner product of the dual operator
    and the primal vector
    """
    def __init__(self, tensor_names=None, tensor_elements=None, tensor_coeffs=None, bias=0, scalar=0):
        # DON'T PUT MUTABLE TYPES AS OBJECT INIT PARAMETERS. MUTABLE INPUTS ARE ABSORBED BY ALL FUTURE OBJECT INSTANCES

        # these specify which tensor of MultiTensor to talk to and which elements
        if tensor_names is None:
            self.primal_tensors_names = []
        else:
            self.primal_tensors_names = tensor_names

        if tensor_elements is None:
            self.primal_elements = []
        else:
            self.primal_elements = tensor_elements

        # these specify the coefficients multiplied by each multiTensor element
        if tensor_coeffs is None:
            self.primal_coeffs = []
        else:
            self.primal_coeffs = tensor_coeffs

        # this is the constant bias for each dual element
        self.constant_bias = bias

        # this is the result of the inner product of the dual on the vector space plus the bias
        self.dual_scalar = scalar

    def add_element(self, name, element, coeff):
        """
        Load an element into the data structure
        """
        if not isinstance(element, tuple):
            raise TypeError("element needs to be a tuple of indices")
        if not isinstance(name, str):
            raise TypeError("name needs to be a string")
        if not isinstance(coeff, (float, int)):
            raise TypeError("coeff needs to be a non-complex numerical value")

        self.primal_elements.append(element)
        self.primal_coeffs.append(coeff)
        self.primal_tensors_names.append(name)

    def join_elements(self, other):
        """
        Join two DualBasisElements together to form another DualBasisElement
        :param DualBasisElement other: to join with self
        :return: DualBasisElement
        """
        if not isinstance(other, DualBasisElement):
            raise TypeError("I can only join two DualBasisElements together")

        dbe = DualBasisElement()
        dbe.primal_tensors_names.extend(other.primal_tensors_names)
        dbe.primal_tensors_names.extend(self.primal_tensors_names)

        dbe.primal_elements.extend(other.primal_elements)
        dbe.primal_elements.extend(self.primal_elements)

        dbe.primal_coeffs.extend(other.primal_coeffs)
        dbe.primal_coeffs.extend(self.primal_coeffs)

        dbe.constant_bias += self.constant_bias + other.constant_bias
        dbe.dual_scalar += self.dual_scalar + other.dual_scalar

        return dbe.simplify()

    def simplify(self):
        """
        Mutate the DualBasisElement so that non-unique terms get summed together
        """
        id_dict = {}
        for tname, telement, tcoeff in zip(self.primal_tensors_names, self.primal_elements, self.primal_coeffs):
            id_str = tname + ".".join([str(x) for x in telement])
            if id_str not in id_dict:
                id_dict[id_str] = (tname, telement, tcoeff)
            else:
                id_dict[id_str] = (tname, telement, id_dict[id_str][2] + tcoeff)

        tnames = []
        telements = []
        tcoeffs = []
        for key, el in id_dict.items():
            tnames.append(el[0])
            telements.append(el[1])
            tcoeffs.append(el[2])

        self.primal_coeffs = tcoeffs
        self.primal_tensors_names = tnames
        self.primal_elements = telements

        return self

    def id(self):
        """
        Get the unique string identifier for the dual basis element

        :return: String name
        """
        id_str = ""
        for name, element in zip(self.primal_tensors_names, self.primal_elements):
            id_str += name + ".".join([str(x) for x in element])
        return id_str

    # def __str__(self):
    #     return self.id

    def __iter__(self):
        for t_label, velement, coeff in zip(self.primal_tensors_names, self.primal_elements, self.primal_coeffs):
            yield t_label, velement, coeff

    def __add__(self, other):
        if isinstance(other, DualBasisElement):
            return DualBasis(elements=[self, other])
        elif isinstance(other, DualBasis):
            return other + self
        else:
            raise TypeError("DualBasisElement can be added to same type or DualBasis")


class DualBasis(object):
    """
    A collection of DualBasisElements.  There is no associated order of the elements

    Initial collection of elements is an empty list. Any iterable will be sufficeint.
    """
    def __init__(self, elements=None):

        if elements is None:
            self.elements = []
        else:
            if all(map(lambda x: isinstance(x, DualBasisElement), elements)):
                self.elements = elements
            else:
                raise TypeError("elements must all be DualBasisElement objects")

    def __iter__(self):
        return self.elements.__iter__()

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return len(self.elements)

    def __add__(self, other):
        if isinstance(other, DualBasisElement):
            new_elements = copy.deepcopy(self.elements)
            new_elements.append(other)
            return DualBasis(elements=new_elements)

        elif isinstance(other, DualBasis):
            new_elements = copy.deepcopy(self.elements)
            new_elements.extend(other.elements)
            return DualBasis(elements=new_elements)

        else:
            raise TypeError("DualBasis adds DualBasisElements or DualBasis only")
