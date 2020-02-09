package com.kdtech.mlwithandroid

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.google.firebase.ml.custom.*
import kotlinx.android.synthetic.main.activity_main.*
import java.io.BufferedReader
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val localModel = FirebaseCustomLocalModel.Builder() //for loading model from local assets folder
            .setAssetFilePath("model_unquant.tflite")
            .build()

        val options = FirebaseModelInterpreterOptions.Builder(localModel).build()

        val interpreter = FirebaseModelInterpreter.getInstance(options)

        val inputOutputOptions = FirebaseModelInputOutputOptions.Builder()
            .setInputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 224, 224, 3))
            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, intArrayOf(1, 2)) // here replace 2 with no of class added in your model , for production apps you can read the labels.txt files here and to get no of classes dynamically
            .build()


        /* Here we are using static image from drawable to keep the code minimum and avoid distraction, Recommended method would be to get the image from user by camera or device photos using the same code by handling all this logic in a method and calling that every time */
        val bitmap = Bitmap.createScaledBitmap(
            BitmapFactory.decodeResource(
                resources,
                R.drawable.car2), 224, 224, true)

        val batchNum = 0
        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }
        for (x in 0..223) {
            for (y in 0..223) {
                val pixel = bitmap.getPixel(x, y)
                // Normalize channel values to [-1.0, 1.0]. This requirement varies by
                // model. For example, some models might require values to be normalized
                // to the range [0.0, 1.0] instead.
                input[batchNum][x][y][0] = (Color.red(pixel) - 127) / 255.0f
                input[batchNum][x][y][1] = (Color.green(pixel) - 127) / 255.0f
                input[batchNum][x][y][2] = (Color.blue(pixel) - 127) / 255.0f
            }
        }

        val inputs = FirebaseModelInputs.Builder()
            .add(input) // add() as many input arrays as your model requires
            .build()
        interpreter!!.run(inputs, inputOutputOptions)
            .addOnSuccessListener { result ->
                // ...
                val output = result.getOutput<Array<FloatArray>>(0)
                val probabilities = output[0]
                val reader = BufferedReader(
                    InputStreamReader(assets.open("labels.txt"))
                )
                var higherProbablityFloat = 0F
                for (i in probabilities.indices) {

                    if (higherProbablityFloat<probabilities[i]){
                        val label = reader.readLine()
                        higherProbablityFloat = probabilities[i]
                        tvIdentifiedItem.text = "The Image is of ${label.substring(2)}"
                    }
                }
            }
            .addOnFailureListener { e ->
                // Task failed with an exception
                // ...
                tvIdentifiedItem.text = "exception ${e.message}"
            }
    }
}
