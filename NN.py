from numpy import exp, array, random, dot
class NeuralNetwork():
    def __init__(self):
        #  مولد الأرقام العشوائية ، بحيث يولد نفس الأرقام
        # في كل مرة يتم فيها تشغيل البرنامج.


        random.seed(1)
        # نقوم بنمذجة خلية عصبية واحدة ، مع 3 وصلات إدخال واتصال خرج واحد.
        # نقوم بتعيين أوزان عشوائية لمصفوفة 3 × 1 ، بقيم في النطاق من -1 إلى 1
        # مدى 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

     # تابع السيجمويد الذي يعيد 0 و1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    #الانحدار الخطي من أجل تعديل الأوزان

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # نقوم بتدريب الشبكة العصبية من خلال عملية التجربة والخطأ.
    # ضبط الأوزان الشبكة في كل مرة.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # اجتياز مجموعة التدريب من خلال شبكتنا العصبية (خلية عصبية واحدة).
            output = self.think(training_set_inputs)
            #حساب الفرق بين القيمة المتوقعة والقيمة الحقيقية وهو مايعرف بحساب الخطأ
            error = training_set_outputs - output
            #تعديل الأوزان حسب الانحدار
            # اضرب الخطأ في الإدخال ومرة أخرى بتدرج المنحنى
            # هذا يعني أن الأوزان الأقل ثقة يتم تعديلها بشكل أكبر.
            # هذا يعني أن المدخلات التي تكون صفرية لا تسبب تغيرات في الأوزان.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment



     #الشبكة العصبية
    def think(self, inputs):

        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # مجموعة التدريب. لدينا 4 أمثلة ، كل منها يتكون من 3 قيم إدخال
    # و 1 قيمة الإخراج.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))
