package com.civilscan.nerf3d

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.civilscan.nerf3d", ctx.packageName)
    }
}
