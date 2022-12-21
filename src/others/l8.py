
# 5.4.1 から


from distutils import dep_util


class MulLayer:
    """
    乗算レイヤ
    """
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    """
    加算レイヤ
    """
    def __init__(self):
        pass
    
    def forward(self, x, y):
        """
        順伝播
        """
        out = x + y

        return out

    def backward(self, dout):
        """
        逆伝播
        """
        dx = dout * 1
        dy = dout * 1

        return dx, dy


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
mul_tax_layer = MulLayer()
add_apple_orange_layer = AddLayer()


apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
sum_orange_apple_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(sum_orange_apple_price, tax)

# 順伝播
print(price)


# 逆伝播
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) # x: 650, y: 1.1
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price) # x: 150, y: 3 -> 3.3, 165
dapple, dapple_num = mul_apple_layer.backward(dapple_price) # x: 100, y: 2 -> 2.2, 110

print(dapple_num, dapple, dorange_num, dorange, dtax)


