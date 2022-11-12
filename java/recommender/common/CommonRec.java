package recommender.common;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

public class CommonRec {

    public static final int RMSE = 1;
    public static final int MAE = 2;

    public static String dataSetName;
    public static String rsDataSetName;
    public static ArrayList<RTuple> trainDataSet =  null;
    public static ArrayList<RTuple> testDataSet =  null;


    public static int maxID = 0; //maxID=行数（user的个数）=列数（item的个数）【对称矩阵】

    public static double RSetSize[];

    public static double lambda = 0.005; // 正则化参数
    protected static double gamma = 0.9; // 动量项系数
    public static int maxRound = 500; // 最多训练轮数
    public static int featureDimension = 0; // 特征维数
    public static double minGap = 0;

    public static int delayCount = 10;

    public double  min_Error = 100; // 最小误差值
    public double cacheMinTotalTime = 0;
    public double minTotalTime = 0;
    public int min_Round = 0; // 记录达到最优结果时的最小迭代次数
    public int total_Round = 0;
    public double total_Time = 0;
   
    public static double[][] unknown; //存预测出来原本未知的值
    public static double known = 0;
    public static int num = 0;
    public static double[][] allvalue;//jiarude 
    public static double[][] catchedFeatureMatrix;
    public static double[] catchedBias;
    public static int mappingScale = 1000;
    public static double featureInitMax = 0.004;
    public static double featureInitScale = 0.004;

    // 辅助矩阵
    public static double[][] featureUp;
    public static double[][] featureDown;
    public static double[] biasUp;
    public static double[] biasDown;

    public double[] bias;
    // 特征矩阵
    public double[][] featureMatrix;
    /* 分别记录当前时刻前面两轮更新之后的特征矩阵的值 */
    public  double[][] lastFeatureMatrix;
    public double[][] penultimateFeaMatrix;
    /* 记录item的bias在前面两轮更新过程中的增量 */
    public double[] lastBias;
    public double[] penultimateBias;

    // 存储特征值的路径
    public static String featureSaveDir;
    public static String biasSaveDir;

    public CommonRec() {
        this.initiInstanceFeatures();
    }

    /*
     * 初始化实例特征矩阵
     */
    public void initiInstanceFeatures() {
        // 加1是为了在序号上与ID保持一致
        bias = new double[maxID + 1];
        featureMatrix = new double[maxID + 1][featureDimension];

        lastFeatureMatrix = new double[maxID + 1][featureDimension];
        penultimateFeaMatrix = new double[maxID + 1][featureDimension];
        lastBias = new double[maxID + 1];
        penultimateBias = new double[maxID + 1];
        allvalue = new double[maxID+1][maxID+1];//自己加的

        for (int i = 1; i <= maxID; i++) {

            bias[i] = catchedBias[i];

            for (int j = 0; j < featureDimension; j++) {
                featureMatrix[i][j] = catchedFeatureMatrix[i][j];
            }
        }
    }

    /*
     * 生成初始的训练集、测试集以及统计各个结点的评分数目
     */
    public static void dataLoad(String dataSetName,String trainFileName, String testFileName, String separator) throws IOException {

        //生成初始的训练集
        trainDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, trainFileName, trainDataSet);

        //生成初始的验证集or测试集
        testDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, testFileName, testDataSet);

        if ("AT".equals(dataSetName)) {
            maxID = 756;
        }
        else if ("EC".equals(dataSetName)){
            maxID = 589;
        }
        else if ("SP".equals(dataSetName)){
            maxID = 904;
        }
        else if ("yeast".equals(dataSetName)){
            maxID = 2497;
        }
        else if ("Human".equals(dataSetName)){
            maxID = 13726;
        }
        else if ("human_57".equals(dataSetName)){
            maxID = 2835;
        }
        else if ("yeast_57".equals(dataSetName)){
            maxID = 2530;
        }else if ("H.pylori_57".equals(dataSetName)){
            maxID = 1428;
        }else if ("C.elegan_56".equals(dataSetName)){
            maxID = 1734;
        }else if ("Drosophila_56".equals(dataSetName)){
            maxID = 5624;
        }else if ("E.coli_56".equals(dataSetName)){
            maxID = 1528;
        }else if ("Hprd_56".equals(dataSetName)){
            maxID = 9463;
        }else if ("Human_56".equals(dataSetName)) {
            maxID = 7803;
        }


        initiRatingSetSize();
    }

    /*
     * 数据集生成器
     */
    public static void dataSetGenerator(String separator, String fileName, ArrayList<RTuple> dataSet) throws IOException {

        File fileSource = new File(fileName);
        BufferedReader in = new BufferedReader(new FileReader(fileSource));

        String line;
        while (((line = in.readLine()) != null)){
            StringTokenizer st = new StringTokenizer(line, separator);
            String personID = null;
            if (st.hasMoreTokens())
                personID = st.nextToken();
            String movieID = null;
            if (st.hasMoreTokens())
                movieID = st.nextToken();
            String personRating = null;
            if (st.hasMoreTokens())
                personRating = st.nextToken();
            int iUserID = Integer.valueOf(personID);
            int iItemID = Integer.valueOf(movieID);

            // 记录下最大的itemid和userid；因为itemid和userid是连续的，所以最大的itemid和userid也代表了各自的数目
            maxID = (maxID > iUserID) ? maxID : iUserID;
            maxID = (maxID > iItemID) ? maxID : iItemID;
            double dRating = Double.valueOf(personRating);

            RTuple temp1 = new RTuple();
            temp1.userID = iUserID;
            temp1.itemID = iItemID;
            temp1.rating = dRating;
            dataSet.add(temp1);

//            if(iUserID != iItemID){
//                // 不是对角线元素的话，则读入对称项
//                RTuple temp2 = new RTuple();
//                temp2.userID = iItemID;
//                temp2.itemID = iUserID;
//                temp2.rating = dRating*0.1;
//                dataSet.add(temp2);
//            }
        }
        in.close();
    }

    /*
     * 统计各个结点的评分数目
     */
    public static void initiRatingSetSize() {
        RSetSize = new double[maxID + 1];
        for (int i = 0; i <= maxID; i++)
            RSetSize[i] = 0;
        for (RTuple tempRating : trainDataSet) {
            RSetSize[tempRating.userID] += 1;
            RSetSize[tempRating.itemID] += 1;
        }
    }

    /*
     * 声明辅助矩阵，并用随机数进行初始化
     */
    public static void initiStaticFeatures() throws IOException {

        // 加1是为了在序号上与ID保持一致
        catchedFeatureMatrix = new double[maxID + 1][featureDimension];
        catchedBias = new double[maxID + 1];

        featureSaveDir = featureSaveDir + featureDimension + ".txt";
        biasSaveDir = biasSaveDir + featureDimension + ".txt";

        File featureFile = new File(featureSaveDir);        // new File(".") 表示用当前路径 生成一个File实例!!!并不是表达创建一个 . 文件
        File biasFile = new File(biasSaveDir);

        if(featureFile.exists() && biasFile.exists()) { // 如果由指定路径下的文件或目录存在则返回 TRUE，否则返回 FALSE。
            System.out.println("准备读取指定初始值...");
            readFeatures(catchedFeatureMatrix, featureSaveDir);  // 读取特征矩阵
            readBias(catchedBias, biasSaveDir);
            System.out.println("读取完毕！！！");
        }else{
            System.out.println("准备生成随机初始值...");
            // 初始化特征矩阵,采用随机值,从而形成一个K阶逼近
            Random random = new Random(System.currentTimeMillis());
            for (int i = 1; i <= maxID; i++) {

                int tempB = random.nextInt(mappingScale);
                catchedBias[i] = featureInitMax - featureInitScale * tempB / mappingScale;

                // 特征矩阵的初始值在(0,0.004]
                for (int j = 0; j < featureDimension; j++) {
                    int temp = random.nextInt(mappingScale); //返回[0,mappingScale)随机整数
                    catchedFeatureMatrix[i][j] = featureInitMax - featureInitScale * temp / mappingScale;
                }
            }

            // 写入文件
            writeFeatures(catchedFeatureMatrix,featureSaveDir);
            writeBias(catchedBias,biasSaveDir);
            System.out.println("写入随机初始值完毕！！！");
        }
        // 声明辅助矩阵
        initiAuxArray();
    }

    //private static void writeFeatures(double[][] catchedFeatureMatrix, String featureSaveDir) throws IOException {
    public static void writeFeatures(double[][] catchedFeatureMatrix, String featureSaveDir) throws IOException {
        FileWriter fw = new FileWriter(featureSaveDir);
        // FileWriter fw = new FileWriter(new File(featureSaveDir));

        for(int i = 1; i <= maxID; i++) {
            for(int k = 0; k < featureDimension; k++) {
                fw.write(catchedFeatureMatrix[i][k] + "::");
            }
            fw.write("\n");
        }
        fw.flush();
        fw.close();
    }

    //private static void writeBias(double[] catchedBias, String biasSaveDir) throws IOException {
    public static void writeBias(double[] catchedBias, String biasSaveDir) throws IOException {

        FileWriter fw = new FileWriter(biasSaveDir);

        for(int i = 1; i <= maxID; i++) {
            fw.write(catchedBias[i] + "::");
        }
        fw.flush();
        fw.close();
    }

    private static void readFeatures(double[][] catchedFeatureMatrix, String featureSaveDir) throws IOException {

        BufferedReader in = new BufferedReader(new FileReader(featureSaveDir));
        String line;  // 一行数据
        int i = 1;    // 行标
        while((line = in.readLine()) != null){
            String [] temp = line.split("::"); // 各数字之间用"::"间隔
            for(int k = 0; k < featureDimension; k++) {
                catchedFeatureMatrix[i][k] = Double.valueOf(temp[k]);
            }
            i++;
        }
        in.close();
    }

    private static void readBias(double[] catchedBias, String biasSaveDir) throws IOException {

        BufferedReader in = new BufferedReader(new FileReader(biasSaveDir));
        String line;  // 一行数据
        if((line = in.readLine()) != null){
            String [] temp = line.split("::"); // 各数字之间用"::"间隔
            for(int j = 0; j < maxID; j++) {
                catchedBias[j + 1] = Double.valueOf(temp[j]);
            }
        }
        in.close();
    }

    /*
     * 声明辅助矩阵
     */
    public static void initiAuxArray() {

        // 加1是为了在序号上与ID保持一致
        featureUp = new double[maxID + 1][featureDimension];
        featureDown = new double[maxID + 1][featureDimension];
        biasUp = new double[maxID + 1];
        biasDown = new double[maxID + 1];
    }

    /*
     * 将辅助矩阵的元素置为0
     */
    public static void resetAuxArray() {
        for (int i = 1; i <= maxID; i++) {

            biasUp[i] = 0;
            biasDown[i] = 0;

            for (int j = 0; j < featureDimension; j++) {
                featureUp[i][j] = 0;
                featureDown[i][j] = 0;
            }
        }
    }

    /*
     * 计算考虑线性偏差的预测值
     */
    public double getPrediction(int userID, int itemID, boolean ifBias) {
        double ratingHat = 0;
        ratingHat += dotMultiply(featureMatrix[userID], featureMatrix[itemID]);
        if (ifBias){
            ratingHat += bias[userID] + bias[itemID];
        }
        return ratingHat;
    }

    // 计算两个向量点乘
    public static double dotMultiply(double[] x, double[] y) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    public double testRMSE(boolean ifbias) {

        // 计算在测试集上的RMSE
        double sumRMSE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID, ifbias);
            sumRMSE += Math.pow((actualRating - ratinghat), 2);
            sumCount++;
        }
        double RMSE = Math.sqrt(sumRMSE / sumCount);
        return RMSE;
    }

    public double testMAE(boolean ifbias) {
        // 计算在测试集上的MAE
        double sumMAE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID, ifbias);
            sumMAE += Math.abs(actualRating - ratinghat);
            sumCount++;
        }
        double MAE = sumMAE / sumCount;
        return MAE;
    }

    public void outPutModelSym(boolean ifbias) throws IOException {

       FileWriter fw = new FileWriter(
               new File("./" + rsDataSetName + "_ModelSym.txt"), true);
       fw.write("i-j:\n");
       for (RTuple trainR: trainDataSet) {
           double ratinghat_ij = getPrediction(trainR.userID, trainR.itemID, ifbias);
           fw.write(ratinghat_ij + "\n");
           fw.flush();
       }
       for (RTuple trainR: testDataSet) {
           double ratinghat_ij = getPrediction(trainR.userID, trainR.itemID, ifbias);
           fw.write(ratinghat_ij + "\n");
           fw.flush();
       }
       fw.write("\n\nj-i:\n");
       for (RTuple trainR: trainDataSet) {
           double ratinghat_ji = getPrediction(trainR.itemID, trainR.userID, ifbias);
           fw.write(ratinghat_ji + "\n");
           fw.flush();
       }
       for (RTuple trainR: testDataSet) {
           double ratinghat_ji = getPrediction(trainR.itemID, trainR.userID, ifbias);
           fw.write(ratinghat_ji + "\n");
           fw.flush();
       }
       fw.close();
    	
    	FileWriter fwUp = new FileWriter(
                new File("./" + rsDataSetName  + "ifbias=" + ifbias + "_ModelSym_Up.txt"));
        FileWriter fwDown = new FileWriter(
                new File("./" + rsDataSetName  + "ifbias=" + ifbias + "_ModelSym_Down.txt"));
        fwUp.write("i-j:\n");
        fwDown.write("j-i:\n");
        for(int i = 1; i <= 1500; i++){
            for(int j = 1500; j >= i; j--){
                double ratinghatUp = getPrediction(i, j, ifbias);
                double ratinghatDown = getPrediction(j, i, ifbias);
                fwUp.write(ratinghatUp + "\n");
                fwDown.write(ratinghatDown + "\n");
                fwUp.flush();
                fwDown.flush();
            }
        }
        fwUp.close();
        fwDown.close();
    }

    public void outPutModelNonnega() throws IOException {

        FileWriter fw = new FileWriter(
                new File("./" + rsDataSetName + "_ModelNonnega.txt"), true);
        for(int id = 1; id <= maxID; id++) {
            for(int dim = 0; dim < featureDimension; dim++){
                fw.write(featureMatrix[id][dim] + "\n");
                fw.flush();
            }
        }
        fw.close();
    }

    public void printNegativeFeature() {

        System.out.println("************************** negative bias:*****************************");
        for (int i = 1; i <= maxID; i++)
            if(bias[i] < 0)
                System.out.println(bias[i]);

        System.out.println("************************** negative feature:**************************");
        for (int i = 1; i <= maxID; i++)
            for (int j = 0; j < featureDimension; j++)
                if (featureMatrix[i][j] < 0)
                    System.out.println(featureMatrix[i][j]);

        System.out.println("***************************************************************************");

    }

     public void outPutfeaturematrix() throws IOException {

         FileWriter fw = new FileWriter(
                 new File("C:/Users/38457/Desktop/java/feature/" + rsDataSetName + "_featurematrix.txt"));
         for(int id = 1; id <= maxID; id++) {
             for(int dim = 0; dim < featureDimension; dim++){
                 fw.write(featureMatrix[id][dim] +"::");
             }
             fw.write("\n");
         }
         fw.flush();
         fw.close();
     }
}
