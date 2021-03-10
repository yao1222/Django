from django.shortcuts import render
from django.http.response import HttpResponse

import csv
import pandas as pd
import numpy as np

# Create your views here.


def index(request):
    # csv = request.FILES['data_source']
    # df = pd.read_csv(csv, encoding='big5')
    parameters_1 = [request.POST.get('weight_1'), request.POST.get(
        'weight_2'), request.POST.get('weight_3')]
    parameters_2 = [request.POST.get('weight_4'), request.POST.get(
        'criterion_1'), request.POST.get('criterion_2')]
    parameters_3 = [request.POST.get('criterion_3'), request.POST.get(
        'criterion_4'), request.POST.get('id_9')]
    context = {}

    return render(request, 'index.html', context)


# 最後提交計算
def form_submit(request):
    if request.FILES:
        csv = request.FILES['data_source']
        data = pd.read_csv(csv, encoding='UTF-8')
        col_len = len(data.columns)

        # 取出weight
        weight = [request.POST.get('weight_1'), request.POST.get(
            'weight_2'), request.POST.get('weight_3'), request.POST.get('weight_4')]
        weight_copy = weight.copy()
        for i in weight_copy:
            if i == '':
                weight.remove(i)
            else:
                pass
        # 判斷weight欄位數量相符
        if len(weight) == (col_len-1):
            pass
        else:
            weight_error = 'Step3中，您輸入權重的數量與上傳檔案的欄位不相符!'
            context = {
                'weight_error': weight_error
            }
            return render(request, 'error.html', context)

        myweight = np.array(weight, dtype=np.float32)

        # 取出criterion
        criterion = [request.POST.get(
            'criterion_1'), request.POST.get('criterion_2'), request.POST.get('criterion_3'), request.POST.get('criterion_4')]
        criterion_copy = criterion.copy()
        for i in criterion_copy:
            if i == 'NA':
                criterion.remove(i)
            else:
                pass
        # 判斷criterion欄位數量相符
        if len(criterion) == (col_len-1):
            pass
        else:
            criterion_error = 'Step4中，您輸入criterion數量與上傳檔案的欄位不相符，請重新輸入!'
            context = {
                'criterion_error': criterion_error
            }
            return render(request, 'error.html', context)

        myweight = np.array(weight, dtype=np.float32)
        mycriterion = np.array(criterion, dtype=np.float32)

        # Create an empty list
        Row_list = []
        Name_list = []
        my_list = []
        count_col = data.shape[1]  # 總共幾行(包括第一行的名稱)
        col_name = list(data.columns[1:])  # column名稱(不包含第一欄名字)

        # Iterate over each row
        for index, rows in data.iterrows():
            # Create list for the current row
            if my_list == []:
                for j in range(1, count_col, 1):
                    my_list.append(rows[j])
            else:
                my_list = []
                for j in range(1, count_col, 1):
                    my_list.append(rows[j])
            #my_list =[rows[1], rows[2], rows[3], rows[4]]

            # append the list to the final list
            Row_list.append(my_list)
            Name_list.append(rows[0])

        C = None
        optimum_choice = None

        a = Row_list
        w = myweight
        I = mycriterion

        # Decision Matrix
        a = np.array(a, dtype=np.float).T
        assert len(a.shape) == 2, "Decision matrix a must be 2D"

        # Number of alternatives, aspects
        (n, J) = a.shape

        # Weight matrix
        w = np.array(w, dtype=np.float)
        assert len(w.shape) == 1, "Weights array must be 1D"
        assert w.size == n, "Weights array wrong length, " + \
            "should be of length {}".format(n)

        # Benefit (True) or Cost (False) criteria?
        I = np.array(I, dtype=np.int8)
        assert len(I.shape) == 1, "Criterion array must be 1D"
        assert len(I) == n, "Criterion array wrong length, " + \
            "should be of length {}".format(n)

        # Initialise best/worst alternatives lists
        ab, aw = np.zeros(n), np.zeros(n)

        # def step1():
        """ TOPSIS Step 1
        Calculate the normalised decision matrix (self.r)
        """
        r = a/np.array(np.linalg.norm(a, axis=1)[:, np.newaxis])

        # def step2(self):
        """ TOPSIS Step 2
        Calculate the weighted normalised decision matrix
        Two transposes required so that indices are multiplied correctly:
        """
        v = (w * r.T).T

        # def step3(self):
        """ TOPSIS Step 3
        Determine the ideal and negative-ideal solutions
        I[i] defines i as a member of the benefit criteria (True) or the cost
        criteria (False)
        """
        # Calcualte ideal/negative ideals
        ab = np.max(v, axis=1) * I + np.min(v, axis=1) * (1 - I)
        aw = np.max(v, axis=1) * (1 - I) + np.min(v, axis=1) * I

        # def step4(self):
        """ TOPSIS Step 4
        Calculate the separation measures, n-dimensional Euclidean distance
        """
        # Create two n long arrays containing Eculidean distances
        # Save the ideal and negative-ideal solutions
        db = np.linalg.norm(v - ab[:, np.newaxis], axis=0)
        dw = np.linalg.norm(v - aw[:, np.newaxis], axis=0)

        # def step5(self):
        """ TOPSIS Step 5 & 6
        Calculate the relative closeness to the ideal solution, then rank the
        preference order
        """
        # Ignore division by zero errors
        # np.seterr(all='ignore')
        # Find relative closeness
        C = dw / (dw + db)
        get_length = len(C)

        # --------好的-----------
        optimum_choice = C.argsort()[-1]
        count_good = 0
        opt_idx = []
        dict_good = {'name': [], 'raw': []}
        for i in range(get_length):
            if C[i] == C[optimum_choice]:
                count_good += 1
                opt_idx.append(i)

        if count_good == 0:
            opt_idx.append(optimum_choice)

        # --------不好的-----------
        bad_choice = C.argsort()[0]
        count_bad = 0
        bad_idx = []
        dict_bad = {'name': [], 'raw': []}
        for j in range(get_length):
            if C[j] == C[bad_choice]:
                count_bad += 1
                bad_idx.append(j)

        if count_bad == 0:
            bad_idx.append(bad_choice)

        # print('--------好的-----------')
        for idx in opt_idx:
            dict_good['name'].append(Name_list[idx])
            dict_good['raw'].append(a[:, idx])

        # print('--------不好的-----------')
        for idx in bad_idx:
            dict_bad['name'].append(Name_list[idx])
            dict_bad['raw'].append(a[:, idx])

    context = {
        'csvData': data,
        'myweight': myweight,
        'mycriterion': mycriterion,
        'opt_idx': opt_idx,
        'bad_idx': bad_idx,
        'dict_bad': dict_bad,
        'dict_good': dict_good,
        'col_name': col_name
    }
    return render(request, 'result.html', context)


def test(request):
    return render(request, 'test.html')
