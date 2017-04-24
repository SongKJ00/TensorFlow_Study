logistic classification이란?

logistic classification이란 말 그대로 논리적인 분류이다. 
형성되는 그래프의 y값은 0과 1 사이에 있어 binary function, 또 그래프의 모양이 S자를 닮았다하여 sigmoid function이라고도 한다. 
sigmoid function에서는 y값을 0과 1의 중간값 0.5를 기준으로 0.5이상이면 1로, 아니면 0으로 생각한다(True or False).

우리는 이전에 linear regression을 공부했을 때 cost값을 줄이기 위해 Gradient Descent Algorithm을 사용했다. 
W와 cost(W)의 그래프가 이차함수를 그렸기 때문에 가능했다.

하지만 sigmoid function은 함수식 자체가 W와 cost(W)의 그래프를 이차함수로 그리지 못하고 약간 울퉁불퉁한 이차함수를 그리게 된다. 
따라서 Gradient Descent Algorithm를 바로 사용하지는 못한다.

linear regression에서는 cost(W)값을 각 hypothesis와 Y의 오차의 제곱의 평균으로 구했다.
여기서는 Y값이 0과 1, 단 두가지로 정해져 있기 때문에 조금은 다른 방법으로 cost(W)값을 구한다.

Y값이 0과 1 중 하나이므로 hypothesis도 0과 1 중 하나이다(실제로는 0<=y<=1범위에서 y는 모든 값을 가진다. 0과 1만이 아니라. )

이때의 cost값은 log함수를 통해 구할 수 있다.
cost(W) = -ylog(H(x))-(1-y)(log((1-H(x))))
y = 1일 때 -log(H(x))함수는 H(x) = 1일 때 0, H(x) = 0일 때, 무한대의 값을 갖는다. 
즉, H(x) = 1이면 cost(W) = 0, H(x) = 0이면 cost(W) = Infinite이다.

y = 0일 때 -log(1-H(x))함수는 H(x) = 1일 때 무한대, H(x) = 0일 때 0의 값을 갖는다.
즉, H(x) = 1이면 cost(W) = Infinite, H(x) = 0이면 cost(W) = 0이다.

이렇게 모든 지점에서의 오차의 평균을 구하면
cost(W) = -tf.reduce_mean(y*log(H(x)+(1-y)*log(1-H(x))))
y = 1일 때는 y*log(H(x))부분만 값이 도출되고((y-1)*log(1-H(x))=0), y = 0일 때는 (1-y)*log(1-(H(x)))부분만 값이 도출된다(y*log(H(x))=0).

이렇게 모든 지점에서의 hypothesis와 y의 오차의 평균 cost(W)을 구할 수 있다. 
따라서, 이제는 W와 cost(W)로 이루어진 이차함수 그래프가 생성된다.

여기서 이제 linear regression에서 사용했던 Gradient Descent Algorithm을 사용하여 최적의 W값을 도출하면 된다.
그러면 그렇게 도출된 W값으로 hypothesis을 도출해 실제 y와 비교하여 정확도를 확인할 수가 있게 된다.

이렇게 sigmoid function으로 나타나는 데이터들을 hypothesis와 y의 오차의 평균을 통해 예측한 데이터와 실제` 데이터와의 정확성을 확인할 수 있다.


