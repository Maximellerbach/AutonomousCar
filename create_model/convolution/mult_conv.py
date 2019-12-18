

class calc_opp():
    def __init__(self):
        self.tot_mult = 0

    def conv2d(self, res=(120, 160, 3), kernel_size=(3,3), strides=(2,2), nfilter=4):
        mult = 0
        pad = (res[0]-kernel_size[0]//2, res[1]-kernel_size[1]//2)
        nres = (pad[0]//strides[0], pad[1]//strides[1], nfilter)
        for _ in range(0, pad[0], strides[0]):
            for _ in range(0, pad[1], strides[1]):
                mult += kernel_size[0]*kernel_size[1]*res[2]*nfilter

        self.tot_mult += mult
        return mult, nres


    def dense(self, dim=100, ndim=50, bias=0):
        mult = dim*(ndim+bias)
        self.tot_mult += mult
        return mult, ndim

def model_tot():
    mult = calc_opp()

    # m, res = mult.conv2d(kernel_size=(5,5), strides=(2,2), res=(128, 128, 3), nfilter=16)
    # m, res = mult.conv2d(kernel_size=(3,3), strides=(2,2), res=res, nfilter=24)
    # m, res = mult.conv2d(kernel_size=(3,3), strides=(2,2), res=res, nfilter=48)
    # m, res = mult.conv2d(kernel_size=(3,3), strides=(2,2), res=res, nfilter=96)
    # m, res = mult.conv2d(kernel_size=(3,3), strides=(2,2), res=res, nfilter=128)
    

    m, res = mult.conv2d(kernel_size=(5,5), strides=(3,3), res=(128, 128, 3), nfilter=8)
    m, res = mult.conv2d(kernel_size=(3,3), strides=(2,2), res=res, nfilter=16)
    m, res = mult.conv2d(kernel_size=(3,3), strides=(2,2), res=res, nfilter=32)
    m, res = mult.conv2d(kernel_size=(3,3), strides=(2,2), res=res, nfilter=48)
    m, res = mult.conv2d(kernel_size=(3,3), strides=(2,2), res=res, nfilter=256)
    print(res)

    dim = res[0]*res[1]*res[2]
    m, dim = mult.dense(dim=dim, ndim=128)
    m, dim = mult.dense(dim=dim, ndim=64)
    m, dim = mult.dense(dim=dim, ndim=18)

    return mult.tot_mult


if __name__ == "__main__":
    t = model_tot()
    print(t)