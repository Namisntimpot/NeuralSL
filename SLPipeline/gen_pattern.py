import os
import numpy as np
import cv2

# 只考虑黑白...
class PatternGenerator:
    def __init__(self, width, height, n_patterns = None, output_dir = None, format = 'png', save_codemat_same_dir = False) -> None:
        self.width = width
        self.height = height
        self.n_patterns = n_patterns
        self.output_dir = output_dir
        self.format = format
        self.save_codemat_same_dir = save_codemat_same_dir
        
    def gen_pattern(self, save = True):
        '''
        返回元组, 分别是pattern的数组和 code matrix
        '''
        if save:
            assert self.output_dir is not None

    def codematrix2patterns(self, codemat:np.ndarray):
        mat = np.expand_dims(codemat, axis=1)  # (n_patterns, 1, w)
        return mat.repeat(self.height, 1)

    def save_all_to_dir(self, patterns:list[np.ndarray], codematrix:np.ndarray):
        for i in range(len(patterns)):
            pat = patterns[i]
            self.save_to_dir(pat, i)
        self.save_to_dir(codematrix, 'code_matrix')

    def save_to_dir(self, img:np.ndarray, index:int | str):
        '''
        认为img在0-1范围内.
        index为str表示是code matrix
        '''
        if not self.format in ['tif', 'exr']:     # 列举一些浮点数图片格式
            normalized = (img * 255).astype(np.uint8)
        else:
            normalized = img
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if type(index) == str:
            if self.save_codemat_same_dir:
                codedir = self.output_dir
            else:
                codedir = os.path.dirname(self.output_dir)
            cv2.imwrite(os.path.join(codedir, "code_matrix."+self.format), normalized)
        else:
            cv2.imwrite(os.path.join(self.output_dir, "%04d."%(index) + self.format), normalized)


class SinPhaseShiftPattern(PatternGenerator):
    def __init__(self, width, height, n_patterns=None, output_dir=None, format='png',minF = 1, maxF = 8, n_shifts = 4) -> None:
        super().__init__(width, height, (maxF - minF + 1) * n_shifts, output_dir, format)
        self.minF = minF
        self.maxF = maxF
        self.n_shifts = n_shifts


    def gen_pattern(self, save=True):
        super().gen_pattern(save)
        frequencies = np.arange(self.minF, self.maxF + 1)  # 频率从1到16
        phase_shifts = np.linspace(0, 2 * np.pi, self.n_shifts, endpoint=False)  # 10次相位移动
        patterns = []
        code_matrix = np.zeros((self.n_patterns, self.width))
        for freq in frequencies:
            for phase in phase_shifts:
                x = np.linspace(0, 2 * np.pi * freq, self.width)
                y = (1 + np.cos(x + phase)) / 2  # 正弦波图案
                pattern = np.tile(y, (self.height, 1))  # 重复图案
                code_matrix[len(patterns), :] = y
                patterns.append(pattern)
        if save:
            self.save_all_to_dir(patterns, code_matrix)
        return patterns, code_matrix
    

def main():
    sin_pattern = SinPhaseShiftPattern(800, 600, output_dir="./Alacarte/testimg/patterns", format="png", minF=4, maxF=4, n_shifts=4)
    patterns, code_matrix = sin_pattern.gen_pattern()
    print(len(patterns))
    print(code_matrix.shape)


if __name__ == '__main__':
    main()