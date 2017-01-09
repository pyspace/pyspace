"""
Define parameter distributions for pySPACE nodes.
"""

from __future__ import division

import abc
import copy
import numpy
from scipy import stats

try:
    # noinspection PyPackageRequirements
    from matplotlib import pyplot as plt
    from matplotlib import ticker as mtick
except ImportError:
    plt = None
    mtick = None


PARAMETER_ATTRIBUTE = "__hyperparameters"


class ParameterDecorator(object):
    """
    Abstract base class for creating parameter decorators
    to declare the as optimization parameters.

    BE CAREFUL WHEN IMPLEMENTING NEW DECORATORS. THEY MUST BE
    WRAPPED AND SUPPORTED BY __ALL__ OPTIMIZATION ALGORITHMS.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, parameter_name):
        """
        Create a new optimization parameter with `parameter_name` as name

        The names of parameters have to be unique, as they get identified
        by the name.

        :param parameter_name: The name of the parameter to create.
        :type parameter_name: str
        """
        self.parameter_name = parameter_name

    @abc.abstractmethod
    def execute(self, class_, parameters):
        """
        Execute the decorator

        This method will be called during creation of the class object
        and will update the given set of hyperparameters according
        to the implementation of the subclass.

        :param class_: The class object this parameter decorates
        :type class_: type
        :param parameters: The set of parameters to append to or delete from
        :type parameters: set(ParameterDecorator)
        """
        raise NotImplementedError("Execute Method has to be overwritten by subclasses")

    @abc.abstractmethod
    def plot(self):
        """
        Plot the given parameter distribution

        This method is used to plot the specified distribution of this decorator.
        For plotting either (if installed) the `matplotlib.pyplot` can be used and return a figure,
        or (if not installed) a string specifying the distribution can be returned.

        :returns: The figure where this distribution is plotted or a string specifying the distribution
        :rtype: matplotlib.figure.Figure | str
        """
        raise NotImplementedError()

    def __call__(self, class_):
        if not hasattr(class_, PARAMETER_ATTRIBUTE):
            # No hyper parameter attribute, create a new one
            setattr(class_, PARAMETER_ATTRIBUTE, set())
        # Deep copy the parameter attribute to avoid side-effects to super-classes
        parameters = copy.deepcopy(getattr(class_, PARAMETER_ATTRIBUTE))
        # Execute the Decorator on the copy
        self.execute(class_, parameters)
        # And replace the attribute with the copy
        setattr(class_, PARAMETER_ATTRIBUTE, parameters)
        # Return the class object
        return class_

    def __eq__(self, other):
        if hasattr(other, "parameter_name"):
            return self.parameter_name == other.parameter_name
        elif isinstance(other, basestring):
            return self.parameter_name == other
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.parameter_name)

    def __str__(self):
        return self.parameter_name

    def __repr__(self):
        return "{cls}<{name}>".format(cls=self.__class__.__name__, name=self.parameter_name)


class AddedParameterDecorator(ParameterDecorator):
    """
    This mixin adds an execute method to classes that inherit from it.


    This mixin will check if the given parameter is already defined
    and if so removes the old definition and adds the new one.
    """

    __metaclass__ = abc.ABCMeta

    def execute(self, class_, parameters):
        if self in parameters:
            parameters.remove(self)
        parameters.add(self)


class QMixin(object):
    """
    This mixin adds a regulation parameter `q`
    to bind a distribution to discrete values.
    """

    def __init__(self, q):
        """
        Add the new regulation parameter.

        :param q: The regulation value
        :type q: float
        """
        self.__q = q

    @property
    def q(self):
        return self.__q

    def round_to_q(self, value):
        """
        Round a value to the next multiple of `q`.

        This is required for the plotting only.

        :param value: The value to round
        :type value: float
        """
        return numpy.round(value / self.q) * self.q

    def calc_probability(self, value, pdf_func):
        # Integral borders
        a = value - self.q / 2
        b = value + self.q / 2
        # Create a linear space between these two borders
        x, dx = numpy.linspace(start=a, stop=b, num=10000, retstep=True)
        # Then calculate the PDF value for each of them
        y = numpy.vectorize(lambda v: pdf_func(v))(x)
        # And then integrate
        return numpy.trapz(y=y, x=x, dx=dx)


class ChoiceParameter(AddedParameterDecorator):
    """
    Defines a parameter as to be chosen from
    the given set of options.

    This parameter will be then chosen from this
    set during the optimization.

    ..code-block:: python

       >>> @ChoiceParameter("test", choices=["A", "B", "C"]
       ... class A(object):
       ...     pass
    """

    def __init__(self, parameter_name, choices):
        """
        Create a new choice parameter with `parameter_name` as name
        and `choices` as the possible values.

        :param parameter_name: The name of the parameter to create.
        :type parameter_name: str
        :param choices: The possible values for this parameter.
        :type choices: str | List[str]
        """
        super(ChoiceParameter, self).__init__(parameter_name=parameter_name)
        if not isinstance(choices, list):
            choices = [choices]
        self.__choices = choices

    @property
    def choices(self):
        return self.__choices

    def plot(self):
        return "{self!r}(choices={self.choices!s})".format(self=self)


class BooleanParameter(ChoiceParameter):
    """
    Defines a parameter as a being a boolean parameter.

    This parameter will either be "true" or "false"
    during the optimization.


    ..code-block:: python

       >>> @BooleanParameter("test")
       ... class A(object):
       ...     pass
    """

    def __init__(self, parameter_name):
        """
        Creates a new boolean parameter with `parameter_name` as name.

        :param parameter_name: The name of the parameter to create.
        :type parameter_name: str
        """
        super(BooleanParameter, self).__init__(parameter_name, [True, False])


class PChoiceParameter(ChoiceParameter):
    """
    Defines a parameter as a probability choice.

    This parameter will sample each of the given
    choices with the according probability.


    ..code-block:: python

       >>> @PChoiceParameter("test", choices={"A": 0.5, "B": 0.25, "C": 0.25})
       ... class A(object):
       ...     pass
    """

    def __init__(self, parameter_name, choices):
        """
        Create a new probability choice parameter
        with `parameter_name` as name and `choices`
        as the possible values.

        Each choice must be a tuple containing first
        the probability in range from 0 to 1 for that
        choice and the value to choose as a second argument.
        The probabilities of all choices need to sum up to 1.

        :param parameter_name: The name of the parameter to create.
        :type parameter_name: str
        :param choices: A dictionary of tuples containing the value
                        of each choice as keys and the corresponding
                        probabilities as values.
        :type choices: dict[object, float]
        """
        if sum(choices.values()) != 1:
            raise RuntimeError("The probabilities for parameter '%s'"
                               "do not sum up to 1" % parameter_name)
        super(PChoiceParameter, self).__init__(parameter_name, choices.items())


class NormalParameter(AddedParameterDecorator):
    """
    Defines a parameter as being normal distributed.

    A normal distributed parameter will be sampled from
    the defined distribution by the mean and standard deviation.

    This parameter will be sampled from a function like:

        normal(mu, sigma)

    This parameter is unbound.

    .. code-block:: python

        >>> @NormalParameter("test", mu=0, sigma=1)
        ... class A(object):
        ...     pass
    """

    class Normal(object):
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma
           
        def pdf(self, x):
            sqrt = numpy.sqrt(2 * numpy.pi * (self.sigma ** 2))
            return 1 / sqrt * numpy.e ** (-((x - self.mu) ** 2) / (2 * self.sigma ** 2))

        def mean(self):
            return self.mu

    def __init__(self, parameter_name, mu, sigma):
        """
        Create a new normal distributed parameter with `parameter_name` as name.

        The mean `mu` and standard deviation `sigma` are defining the distribution
        this parameter will be sampled from.

        :param parameter_name: The name of the parameter to create.
        :type parameter_name: str
        :param mu: The mean value of the distribution for this parameter
        :type mu: float
        :param sigma: The standard deviation of the distribution for this parameter
        :type sigma: float
        """
        super(NormalParameter, self).__init__(parameter_name=parameter_name)
        self.__mu = mu
        self.__sigma = sigma

    @property
    def mu(self):
        return self.__mu

    @property
    def sigma(self):
        return self.__sigma

    def plot(self):
        if plt is not None:
            rv = self.Normal(self.mu, self.sigma)
            # mu +/- 3sigma => 99,7% confidence interval
            start = self.mu - 3 * self.sigma
            stop = self.mu + 3 * self.sigma
            # Min 1.000 samples, max 10.000
            num = min(max(stop - start * 100, 1000), 10000)
            figure = plt.figure()
            plt.ylabel("PDF(x)")
            plt.xlabel("X")
            figure.suptitle("Normal distribution for Parameter {param!s}".format(param=self.parameter_name))
            x = numpy.linspace(start=start, stop=stop, num=num)
            y = numpy.vectorize(lambda x_: rv.pdf(x_))(x)
            axes = plt.plot(x, y, label="mu={self.mu:g}, sigma={self.sigma:g}".format(self=self))
            mean_x = rv.mean()
            mean_y = rv.pdf(mean_x)
            y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
            plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes[0].get_color(),
                        label="Mean: %g" % mean_x)
            plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
            return figure
        else:
            return "{self!r}(mu={self.mu:g}, sigma={self.sigma:g})".format(self=self)


class UniformParameter(AddedParameterDecorator):
    """
    Defines a parameter as being uniform distributed.

    A uniform distributed parameter will be sampled equally distributed
    between a given minimum and maximum value.

    The value of this parameter will be sampled from a function like:

    .. code-block:: python

        uniform(min, max)

    This parameter is bound to [min, max].

    .. code-block:: python

        >>> @UniformParameter("test", min_value=1, max_value=10)
        ... class A(object):
        ...     pass
    """

    class Uniform(object):
        def __init__(self, a, b):
            self.a = a
            self.b = b
            self.__diff = b - a

        def pdf(self, x):
            if self.a > x or x > self.b:
                return 0.0
            return 1.0 / self.__diff

        def cdf(self, x):
            if x <= self.a:
                return 0.0
            elif x >= self.b:
                return 1.0
            else:
                return (x - self.a) / self.__diff 

        def mean(self):
            return .5 * (self.a + self.b)

    def __init__(self, parameter_name, min_value, max_value):
        """
        Create a new uniform distributed parameter with `parameter_name` as name.

        The `min_value` and `max_value` define the borders for this distribution
        in between which the value will be sampled.

        :param min_value: The minimum value of the parameter
        :type min_value: float
        :param max_value: The maximum value of the parameter
        :type max_value: float
        """
        super(UniformParameter, self).__init__(parameter_name=parameter_name)
        self.__min = min_value
        self.__max = max_value

    @property
    def min(self):
        return self.__min

    @property
    def max(self):
        return self.__max

    def plot(self):
        if plt is not None:
            rv = self.Uniform(self.min, self.max)
            # Min 1.000 samples, max 10.000
            num = min(max(self.max - self.min, 1000), 10000)
            figure = plt.figure()
            plt.ylabel("PDF(x)")
            plt.xlabel("X")
            plt.suptitle("Uniform distribution of parameter {param!s}".format(param=self.parameter_name))
            x = numpy.linspace(start=self.min - 1, stop=self.max + 1, num=num)
            y = numpy.vectorize(lambda value: rv.pdf(value))(x)
            axes = plt.plot(x, y, label="min={self.min:g}, max={self.max:g}".format(self=self))
            mean_x = rv.mean()
            mean_y = rv.pdf(mean_x)
            y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
            plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes[0].get_color(),
                        label="Mean: %g" % mean_x)
            plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
            return figure
        else:
            return "{self!r}(min={self.min:g}, max={self.max:g})".format(self=self)


class QNormalParameter(NormalParameter, QMixin):
    """
    Defines a parameter as being normal distributed but to only
    take discrete values.

    A q-normal distributed parameter will be sampled from the
    distribution defined by a mean and a standard deviation
    but it will be bound to discrete values regularized by a
    regulation parameter.
    Therefore the value will be sampled from a function like this:

        round(normal(mu, sigma) / q) * q

    This parameter is unbound.

    .. code-block:: python

        >>> @QNormalParameter("test", mu=0, sigma=1, q=0.5)
        ... class A(object):
        ...     pass
    """

    def __init__(self, parameter_name, mu, sigma, q):
        """
        Creates a new q-normal distributed parameter with `parameter_name` as name.

        The mean `mu` and standard deviation `sigma` are defining the distribution
        this parameter will be sampled from. But it will be bound to discrete values
        by the regulation parameter `q`.

        :param parameter_name: The name of the parameter to create.
        :type parameter_name: str
        :param mu: The mean value of the distribution for the parameter
        :type mu: float
        :param sigma: The standard deviation of the distribution for the parameter
        :type sigma: float
        :param q: The regulation parameter to bind the values with.
        :type q: float
        """
        super(QNormalParameter, self).__init__(parameter_name=parameter_name, mu=mu, sigma=sigma)
        QMixin.__init__(self, q=q)

    def plot(self):
        # mu +/- 3sigma => 99,7% confidence interval
        if plt is not None:
            rv = self.Normal(self.mu, self.sigma)
            # mu +/- 3sigma => 99,7% confidence interval
            start = self.round_to_q(self.mu - 3 * self.sigma)
            stop = self.round_to_q(self.mu + 3 * self.sigma)
            figure = plt.figure()
            plt.ylabel("P(X=x)")
            plt.xlabel("X")
            figure.suptitle("Q-Normal distribution for Parameter {param!s}".format(param=self.parameter_name))
            x = numpy.arange(start=start, stop=stop + self.q, step=self.q)
            y = numpy.vectorize(lambda x_: self.calc_probability(x_, rv.pdf))(x)
            axes = plt.scatter(x, y, label="mu={self.mu:g}, sigma={self.sigma:g}, q={self.q:g}".format(self=self))
            mean_x = self.round_to_q(rv.mean())
            mean_y = self.calc_probability(mean_x, rv.pdf)
            y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
            plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes.get_facecolor()[0],
                        label="Mean: %g" % mean_x)
            plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
            return figure
        else:
            return "{self!r}(mu={self.mu:g}, sigma={self.sigma:g}, q={self.q:g})".format(self=self)


class QUniformParameter(UniformParameter, QMixin):
    """
    Defines a parameter as being uniform distributed but to take
    only discrete values.

    A uniform distributed parameter will be sampled equally distributed
    between a given minimum and maximum value and will be bound by a
    regulation parameter.

    The value will be sampled from a function like this:

        round(uniform(min, max) / q) * q

    This parameter is bound to [floor(min), tail(max)].

    .. code-block:: python

        >>> @QUniformParameter("test", min_value=0, max_value=10, q=0.5)
        ... class A(object):
        ...     pass
    """
    def __init__(self, parameter_name, min_value, max_value, q):
        """
        Creates a new q-uniform distributed parameter with `parameter_name` as name.

        The `min_value` and `max_value` define the borders for this distribution
        in between which the value will be sampled. But this distribution
        will be bound to discrete values by the regulation parameter `q`.

        :param min_value: The minimum value of the parameter
        :type min_value: float
        :param max_value: The maximum value of the parameter
        :type max_value: float

        :param q: The regulation parameter to bind the values with.
        :type q: float
        """
        super(QUniformParameter, self).__init__(parameter_name=parameter_name, min_value=min_value, max_value=max_value)
        QMixin.__init__(self, q=q)

    def plot(self):
        if plt is not None:
            start = self.round_to_q(self.min) - 2 * self.q
            stop = self.round_to_q(self.max) + 2 * self.q
            rv = self.Uniform(self.min - self.q / 2, self.max + self.q / 2)
            figure = plt.figure()
            plt.ylabel("P(X=x)")
            plt.xlabel("X")
            plt.suptitle("Q-Uniform distribution of parameter {param!s}".format(param=self.parameter_name))
            x = numpy.arange(start=start, stop=stop + self.q, step=self.q)
            y = numpy.vectorize(lambda value: self.calc_probability(value, rv.pdf))(x)
            axes = plt.scatter(x, y, label="min={self.min:g}, max={self.max:g}, q={self.q:g}".format(self=self))
            mean_x = self.round_to_q(rv.mean())
            mean_y = self.calc_probability(mean_x, rv.pdf)
            y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
            plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes.get_facecolor()[0],
                        label="Mean: %g" % mean_x)
            plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
            return figure
        else:
            return "{self!r}(min={self.min:g}, max={self.max:g}, q={self.q:g})".format(self=self)


class LogNormalParameter(AddedParameterDecorator):
    """
    Defines a parameter as being drawn from the exponential
    of a normal distribution.

    A log-normal distributed parameter will be sampled from
    exponential of the defined distribution by a mean
    and a standard deviation.

    The values for this parameter will be sampled from a function like:

        exp(normal(mu, sigma))

    This distribution causes that the logarithm of the samples values
    are being normal distributed.
    This parameter is bound to positive numbers only.

    .. code-block:: python

        >>> @LogNormalParameter("test", shape=1, scale=1)
        ... class A(object):
        ...     pass
    """
    
    def __init__(self, parameter_name, shape, scale):
        super(LogNormalParameter, self).__init__(parameter_name)
        self.__shape = shape
        self.__scale = scale

    @property
    def shape(self):
        return self.__shape

    @property
    def scale(self):
        return self.__scale

    def plot(self):
        if plt is not None:
            rv = stats.lognorm(s=self.shape, scale=self.scale)
            start, stop = rv.interval(0.99)
            # Min 1.000 samples, max 10.000
            num = min(max(stop - start * 100, 1000), 10000)
            figure = plt.figure()
            plt.ylabel("PDF(x)")
            plt.xlabel("X")
            figure.suptitle("Log-Normal distribution for Parameter {param!s}".format(param=self.parameter_name))
            x = numpy.logspace(start=numpy.log(start), stop=numpy.log(stop), base=numpy.e, num=num)
            y = rv.pdf(x)
            axes = plt.plot(x, y, label="shape={self.shape:g}, scale={self.scale:g}".format(self=self))
            mean_x = rv.mean()
            mean_y = rv.pdf(mean_x)
            y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
            plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes[0].get_color(),
                        label="Mean: %g" % mean_x)
            plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
            plt.xscale("log")
            return figure
        else:
            return "{self!r}(shape={self.shape:g}, scale={self.scale:g})".format(self=self)


class LogUniformParameter(UniformParameter):
    """
    Defines a parameter as being drawn from the exponential
    of a uniform distribution.

    A log-uniform distributed parameter will be sampled from
    the exponential of equally distributed values
    between a given minimum and maximum value.

    The values for this parameter will be sampled from a function like:

        exp(uniform(min, max))

    This distribution causes that the logarithm of the samples values
    are being uniform distributed.
    This parameter is bound to [exp(min), exp(max)].

    .. code-block:: python

        >>> @LogUniformParameter("test", min_value=0, max_value=1000)
        ... class A(object):
        ...     pass
    """
    class LogUniform(object):
        log = numpy.log

        def __init__(self, a, b):
            self.__a = a
            self.__al = self.log(a) if a > 0 else self.log(1e-320)
            self.__b = b
            self.__bl = self.log(b) if b > 0 else self.log(1e-320)

        def pdf(self, x):
            if x >= 0 and self.__al <= self.log(x) <= self.__bl:
                return 1 / (x * (self.__bl - self.__al))
            else:
                return 0.0

        def cdf(self, x):
            xl = self.log(x)
            if xl < self.__al:
                return 0.0
            elif xl > self.__bl:
                return 1
            else:
                return (xl - self.__al) / (self.__bl - self.__al)

        def mean(self):
            return (self.__b - self.__a) / (self.__bl - self.__al)

    def plot(self):
        if plt is not None:
            rv = self.LogUniform(self.min, self.max)
            # Min 1.000 samples, max 100.000
            num = min(max(self.max - self.min, 1000), 100000)
            figure = plt.figure()
            plt.ylabel("PDF(x)")
            plt.xlabel("X")
            plt.suptitle("Log-Uniform distribution of parameter {param!s}".format(param=self.parameter_name))
            x = numpy.logspace(start=numpy.log10(max(self.min - 1, 1e-10)), stop=numpy.log10(self.max + 1),
                               base=10, num=num)
            y = numpy.vectorize(lambda value: rv.pdf(value))(x)
            mean_x = rv.mean()
            mean_y = rv.pdf(mean_x)
            line, = plt.plot(x, y, label="min={self.min:g}, max={self.max:g}".format(self=self))
            y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
            plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=line.get_color(),
                        label="Mean: %g" % mean_x)
            plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
            plt.xscale("log", basex=10)
            plt.yscale("log", basey=10)
            return figure
        else:
            return "{self!r}(min={self.min:g}, max={self.max:g})".format(self=self)


class QLogNormalParameter(LogNormalParameter, QMixin):
    """
    Defines a parameter as being drawn from the exponential
    of a normal distribution.

    A log-normal distributed parameter will be sampled from
    exponential of the defined distribution by a mean
    and a standard deviation but it will be bound to discrete
    values regularized by a regulation parameter.

    The values for this parameter will be sampled from a function like:

        round(exp(normal(mu, sigma)) / q) * q

    This distribution causes that the logarithm of the samples values
    are being normal distributed.
    This parameter is bound to positive number only.

    .. code-block:: python

        >>> @QLogNormalParameter("test", shape=1, scale=1, q=0.5)
        ... class A(object):
        ...     pass
    """
    def __init__(self, parameter_name, shape, scale, q):
        LogNormalParameter.__init__(self, parameter_name, shape, scale)
        QMixin.__init__(self, q)

    def plot(self):
        if plt is not None:
            rv = stats.lognorm(s=self.shape, scale=self.scale)
            start, stop = rv.interval(0.99)
            figure = plt.figure()
            plt.ylabel("P(X=x)")
            plt.xlabel("X")
            figure.suptitle("Q-Log-Normal distribution for Parameter {param!s}".format(param=self.parameter_name))
            x = numpy.arange(start=self.round_to_q(start), stop=self.round_to_q(stop) + self.q, step=self.q)
            y = numpy.vectorize(lambda x_: self.calc_probability(x_, rv.pdf))(x)
            axes = plt.scatter(x, y,
                               label="shape={self.shape:g}, scale={self.scale:g}, q={self.q:g}".format(self=self))
            mean_x = self.round_to_q(rv.mean())
            mean_y = self.calc_probability(mean_x, rv.pdf)
            y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
            plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes.get_facecolor()[0],
                        label="Mean: %g" % mean_x)
            plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
            plt.xscale("log")
            return figure
        else:
            return "{self!r}(shape={self.shape:g}, scale={self.scale:g}, q={self.q:g})".format(self=self)


class QLogUniformParameter(LogUniformParameter, QMixin):
    """
    Defines a parameter as being drawn from the exponential
    of a uniform distribution and being bound to discrete values
    only.

    A q-log-uniform distributed parameter will be sampled from
    the exponential of equally distributed values
    between a given minimum and maximum value and will be bound by a
    regulation parameter.

    The values for this parameter will be sampled from a function like:

        round(exp(uniform(min, max)) / q) * q

    This distribution causes that the logarithm of the samples values
    are being uniform distributed.
    This parameter is bound to [exp(min), exp(max)].

    .. code-block:: python

        >>> @QLogUniformParameter("test", min_value=0, max_value=1000, q=0.5)
        ... class A(object):
        ...     pass
    """
    def __init__(self, parameter_name, min_value, max_value, q):
        LogUniformParameter.__init__(self, parameter_name, min_value, max_value)
        QMixin.__init__(self, q)

    def plot(self):
        rv = self.LogUniform(self.min - self.q / 2, self.max + self.q / 2)
        if plt is not None:
            start = self.round_to_q(self.min) - 2 * self.q
            stop = self.round_to_q(self.max) + 2 * self.q
            figure = plt.figure()
            plt.ylabel("P(X=x)")
            plt.xlabel("X")
            plt.suptitle("Q-Log-Uniform distribution of parameter {param!s}".format(param=self.parameter_name))
            x = numpy.arange(start=start, stop=stop + self.q, step=self.q)
            y = numpy.vectorize(lambda value: self.calc_probability(value, rv.pdf))(x)
            mean_x = self.round_to_q(rv.mean())
            mean_y = self.calc_probability(mean_x, rv.pdf)
            axes = plt.scatter(x, y,
                               label="min={self.min:g}, max={self.max:g}, q={self.q:g}".format(self=self))
            y_max = 1.0 / (plt.ylim()[1] - plt.ylim()[0]) * (mean_y - plt.ylim()[0])
            plt.axvline(x=mean_x, ymax=y_max, linestyle="--", color=axes.get_facecolor()[0],
                        label="Mean: %g" % mean_x)
            plt.legend(loc="best", fancybox=True, framealpha=0.2).draggable(True)
            plt.xscale("log", basex=10)
            plt.yscale("log", basey=10)
            return figure
        else:
            return "{self!r}(min={self.min:g}, max={self.max:g}, q={self.q:g})".format(self=self)


class NoOptimizationParameter(AddedParameterDecorator):
    """
    Defines a previously defined parameter as not being
    an optimization parameter at all.

    This decorator can be used in derived classes where
    optimization parameters of base classes are given concrete values
    or don't matter at all.

    .. code-block:: python

        >>> @NoOptimizationParameter("test")
        ... class A(object):
        ...     pass
    """

    def plot(self):
        return repr(self)


PARAMETER_TYPES = {
    "Choice": ChoiceParameter,
    "Boolean": BooleanParameter,
    "PChoice": PChoiceParameter,
    "Normal": NormalParameter,
    "Uniform": UniformParameter,
    "QNormal": QNormalParameter,
    "QUniform": QUniformParameter,
    "LogNormal": LogNormalParameter,
    "LogUniform": LogUniformParameter,
    "QLogNormal": QLogNormalParameter,
    "QLogUniform": QLogUniformParameter,
    "NoOptimization": NoOptimizationParameter
}
